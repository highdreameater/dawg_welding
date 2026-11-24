"""Force-controlled contact task on a curved surface: the torch tip presses on the pipe with a constant normal force
while it sweeps an arc over the pipe top (the surface normal rotates by 2 * theta_max), with the base held rigid.

Phases (time relative to the hand-off to the stance WBC):
    settle -> brace (base down 4 cm) -> unstow (joint space to a hover pose 3 cm off the surface)
    -> hover (task space) -> approach (force mode: moves along -n at <= vn_max until the force builds to F_des)
    -> sweep x passes (force mode along the arc, tool axis kept on the surface normal) -> dwell
    -> lift (position mode, 5 cm off along the normal) -> stow (joint space) -> hold

Hybrid force/motion control (sim/wbc/stance_wbc.py): tangent-plane position + tool-axis direction are motion tasks,
the normal direction is an admittance loop on the measured contact force, and the measured force enters the
floating-base dynamics of the QP.
"""
import mujoco
import numpy as np

from wbc.stance_wbc import Indexer, ik_arm, mat2quat
from world import PIPE_WORK_X, pipe_frame

PHASES = ["settle", "brace", "unstow", "hover", "approach", "sweep", "dwell", "lift", "stow", "hold"]


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cruise(tau, T, Tr):
    """0 -> 1 over T: constant speed with C2-smooth (min-jerk velocity) ramps of length Tr at both ends."""
    vmax = 1.0 / (T - Tr)
    I = lambda s: 2.5 * s**4 - 3 * s**5 + s**6  # integral of min_jerk
    tau = np.clip(tau, 0.0, T)
    if tau < Tr:
        return vmax * Tr * I(tau / Tr)
    if tau > T - Tr:
        return 1.0 - vmax * Tr * I((T - tau) / Tr)
    return vmax * (tau - Tr / 2)


def fd(fun, t, h=1e-3):
    f0, fp, fm = fun(t), fun(t + h), fun(t - h)
    return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2


def tip_contact_force(m, d, geom_id):
    """Contact force on the contact tip: (normal force [N], contact count, world force vector on the tip).
    The tip is frictionless, so the force vector points exactly along the surface normal at the contact."""
    f, total, n, vec = np.zeros(6), 0.0, 0, np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == geom_id or c.geom2 == geom_id:
            mujoco.mj_contactForce(m, d, i, f)
            total += f[0]
            n += 1
            normal = c.frame[:3]  # contact normal, from geom1 towards geom2
            vec += f[0] * normal * (1.0 if c.geom2 == geom_id else -1.0)
    return total, n, vec


def tool_rotation(n, axis):
    """Tool frame: x (tool axis) = -n (pointing into the surface), z = pipe axis."""
    x = -np.asarray(n, float)
    z = np.asarray(axis, float)
    return np.stack([x, np.cross(z, x), z], 1)


class PipeForceTask:
    HOVER = 0.03

    def __init__(self, cfg, ctrl_model, qpos0, pipe, f_des=5.0, theta_max=np.radians(30), passes=2,
                 speed=0.03, t0=0.0, normal_source="model"):
        """pipe: the geometry the robot *believes* (map, or a perception estimate).
        normal_source: "model" - surface normal / arc from `pipe`;
                       "touch" - once in contact, the surface normal is measured from the contact-force direction
                                 and the arc angle is tracked from it (tactile surface following): only the
                                 approach point, the pipe axis and the radius (for speed) come from `pipe`."""
        self.cfg, self.pipe, self.f_des, self.t0 = cfg, pipe, f_des, t0
        self.normal_source, self.n_touch = normal_source, None
        ix = Indexer(ctrl_model, cfg)
        self.ix, self.stow = ix, ix.arm_stow
        # base: hold the hand-off position, level the attitude (keep heading) while bracing 4 cm down
        self.base_pos0 = qpos0[0:3].copy()
        self.base_quat0 = qpos0[3:7].copy()
        R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, self.base_quat0)
        yaw = np.arctan2(R0[3], R0[0])
        self.base_quat_level = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.level_err = np.zeros(3)
        mujoco.mju_subQuat(self.level_err, self.base_quat_level, self.base_quat0)
        self.base_brace = self.base_pos0 - np.array([0, 0, cfg.brace_dz])

        self.s_work = PIPE_WORK_X  # axial position of the arc, from the near end
        self.work_point = pipe.surface_point(self.s_work, 0.0)[0]
        self.theta_max, self.passes = theta_max, passes
        arc = 2 * theta_max * pipe.radius
        self.T_pass = max(3.0, arc / speed)
        # timeline (relative)
        self.T_ph = dict(settle=0.5, brace=1.0, unstow=2.5, hover=0.5, approach=3.0, sweep=passes * self.T_pass,
                         dwell=1.0, lift=1.0, stow=2.5, hold=1.0)
        t, self.phases = 0.0, []
        for p in PHASES:
            self.phases.append((t, p))
            t += self.T_ph[p]
        self.T = t
        self.start = {p: ts for ts, p in self.phases}  # phase -> start time

        # IK for the hover pose above the first arc point, from the braced stance (multi-start, keep the solution
        # furthest from the joint limits), then check that the whole arc is reachable from it
        q_ik = qpos0.copy(); q_ik[0:3] = self.base_brace; q_ik[3:7] = self.base_quat_level
        th0 = self.theta(self.start["approach"])
        p0, n0 = pipe.surface_point(self.s_work, th0)
        rng = np.random.default_rng(0)
        seeds = [np.array(s, float) for s in cfg.ik_seeds]
        seeds += [rng.uniform(0.9 * ix.arm_lo, 0.9 * ix.arm_hi) for _ in range(30)]
        span = ix.arm_hi - ix.arm_lo
        best, best_score = None, -np.inf
        for seed in seeds:
            q, res = ik_arm(ctrl_model, q_ik, ix, p0 + self.HOVER * n0, tool_rotation(n0, pipe.axis), seed=seed,
                            iters=200, axis_only=True)
            margin = np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span)
            if res < 1e-3 and margin > best_score:
                best, best_score = q, margin
        if best is None:
            raise RuntimeError("hover pose above the pipe is unreachable from this stance - dock closer")
        self.q_hover = best
        q_arc, worst = best, 0.0
        for th in np.linspace(-theta_max, theta_max, 7):
            p, n = pipe.surface_point(self.s_work, th)
            q_arc, res = ik_arm(ctrl_model, q_ik, ix, p, tool_rotation(n, pipe.axis), seed=q_arc, axis_only=True)
            worst = max(worst, res)
        self.arc_ik_residual_mm = 1000 * worst
        self.captured = {}
        self.t_contact, self.t_last, self.wait_s = None, None, 0.0
        self.max_wait = 6.0  # [s] the approach keeps searching for the surface at most this much longer

    # ------------------------------------------------------------------ timeline
    def phase(self, t):
        tr, name = t - self.t0, self.phases[0][1]
        for ts, p in self.phases:
            if tr >= ts:
                name = p
        return name

    def theta(self, tr):
        """Arc angle at relative time tr (passes alternate -theta_max -> +theta_max -> ...)."""
        th = self.theta_max
        if tr < self.start["sweep"]:
            return -th
        k, rem = divmod(min(tr - self.start["sweep"], self.T_ph["sweep"] - 1e-9), self.T_pass)
        s = cruise(rem, self.T_pass, min(1.0, 0.25 * self.T_pass))
        return -th + 2 * th * s if int(k) % 2 == 0 else th - 2 * th * s

    def tip_ref(self, tr):
        return self.pipe.surface_point(self.s_work, self.theta(tr))[0]

    # ------------------------------------------------------------------ references for the WBC
    def ref(self, t, tip_pos, arm_q, f_meas, f_vec=None):
        # contact-triggered timeline: the sweep starts only after the tip has been on the surface for 0.5 s; if the
        # approach time runs out first, the task clock pauses (keeps approaching) for up to max_wait seconds
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        if self.phase(t) == "approach":
            if f_meas > 1.0:
                self.t_contact = self.t_contact or t
            if (t - self.t0 > self.start["sweep"] - dt and (self.t_contact is None or t - self.t_contact < 0.5)
                    and self.wait_s < self.max_wait):
                self.t0 += dt
                self.wait_s += dt
        tr = t - self.t0
        ph = self.phase(t)
        n_arm = len(self.stow)
        s_brace = min_jerk((tr - self.start["brace"]) / self.T_ph["brace"])
        qb = self.base_quat0.copy()
        mujoco.mju_quatIntegrate(qb, self.level_err, s_brace)
        r = dict(base_pos=self.base_pos0 + s_brace * (self.base_brace - self.base_pos0), base_quat=qb,
                 arm_q=self.stow, arm_qd=np.zeros(n_arm), arm_qdd=np.zeros(n_arm), w_ee=0.0, w_post=10.0)
        if ph in ("settle", "brace"):
            return r
        if ph == "unstow":
            f = lambda tt: self.stow + min_jerk((tt - self.t0 - self.start["unstow"]) / self.T_ph["unstow"]) * (self.q_hover - self.stow)
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = fd(f, t)
            return r
        if ph == "stow":
            q_s = self.captured.setdefault("stow_q", arm_q.copy())
            f = lambda tt: q_s + min_jerk((tt - self.t0 - self.start["stow"]) / self.T_ph["stow"]) * (self.stow - q_s)
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = fd(f, t)
            return r
        if ph == "hold":
            return r

        th = self.theta(tr)
        p_surf, n = self.pipe.surface_point(self.s_work, th)
        quat = mat2quat(tool_rotation(n, self.pipe.axis))
        task = dict(w_ee=30.0, w_post=1e-4, arm_q=self.q_hover, ee_quat=quat, axis_only=True)
        if ph == "hover":
            task.update(ee_pos=p_surf + self.HOVER * n, ee_vel=np.zeros(3), ee_acc=np.zeros(3))
        elif ph == "lift":
            p_l = self.captured.setdefault("lift_p", tip_pos.copy())
            f = lambda tt: p_l + min_jerk((tt - self.t0 - self.start["lift"]) / self.T_ph["lift"]) * 0.05 * n
            task["ee_pos"], task["ee_vel"], task["ee_acc"] = fd(f, t)
        elif self.normal_source == "touch" and f_vec is not None and (f_meas > 1.0 or self.n_touch is not None):
            # tactile surface following: normal from the contact force, arc angle measured from that normal. If the
            # contact drops, keep the last measured normal (the WBC then re-approaches along it)
            if f_meas > 1.0:
                n_m = f_vec / np.linalg.norm(f_vec)
                self.n_touch = n_m if self.n_touch is None else self.n_touch + 0.05 * (n_m - self.n_touch)
            n = self.n_touch / np.linalg.norm(self.n_touch)
            a = self.pipe.axis
            up, side = pipe_frame(a)
            th_m = np.arctan2(n @ side, n @ up)
            t_dir = np.cross(n, a)                      # direction of increasing theta on the surface
            th_r, dth, ddth = fd(lambda tt: self.theta(tt - self.t0), t)
            r_est = self.pipe.radius
            task.update(ee_quat=mat2quat(tool_rotation(n, a)),
                        # tangential lag + axial offset (the normal direction is force-controlled)
                        ee_pos=tip_pos + r_est * (th_r - th_m) * t_dir + (self.s_work - (tip_pos - self.pipe.end) @ a) * a,
                        ee_vel=r_est * dth * t_dir,
                        ee_acc=r_est * ddth * t_dir - r_est * dth ** 2 * n,  # incl. centripetal feed-forward
                        force=dict(n=n, f_des=self.f_des, f_meas=f_meas))
            self.th_meas = th_m
        else:  # approach / sweep / dwell: force control along n
            p, v, a = fd(lambda tt: self.tip_ref(tt - self.t0), t)
            _, ndot, _ = fd(lambda tt: self.pipe.surface_point(self.s_work, self.theta(tt - self.t0))[1], t)
            task.update(ee_pos=p, ee_vel=v, ee_acc=a, ee_omega=np.cross(n, ndot),
                        force=dict(n=n, f_des=self.f_des, f_meas=f_meas))
        r.update(task)
        return r
