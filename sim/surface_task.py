"""Force-controlled tracing of an unknown freeform surface: scan it with the wrist camera, reconstruct it, then press
the torch tip on it with a constant normal force while following an S-shaped path across it.

Phases (time relative to the hand-off to the stance WBC):
    settle -> brace -> to_scan (joint space: tool down over the panel centre, wrist camera looking at the panel)
    -> scan (hold still; the caller grabs depth frames and calls set_model(...) at the end of this phase)
    -> to_hover (joint space to 3 cm above the path start, along the *reconstructed* normal)
    -> hover -> approach (force mode, contact-triggered) -> sweep x passes -> dwell -> lift -> stow -> hold

The path is an S-curve in the horizontal plane (along the panel's long side, weaving across it); its height, the
surface normal (force direction, tool axis) and the curvature feed-forward all come from the surface model.
Hybrid force/motion control is the same as for the pipe (sim/wbc/stance_wbc.py).
"""
import mujoco
import numpy as np

from force_task import cruise, fd, min_jerk
from wbc.stance_wbc import Indexer, ik_arm, mat2quat

PHASES = ["settle", "brace", "to_scan", "scan", "to_hover", "hover", "approach", "sweep", "dwell", "lift", "stow",
          "hold"]
R_TOOL_DOWN = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)  # tool x down, camera's wide axis along world y


def tool_frame(n, tangent):
    x = -np.asarray(n, float)
    z = np.asarray(tangent, float) - (np.asarray(tangent, float) @ x) * x
    z /= np.linalg.norm(z) + 1e-12
    return np.stack([x, np.cross(z, x), z], 1)


class SurfaceTraceTask:
    HOVER = 0.03
    MIN_MARGIN = 0.12  # [rad] joint-limit headroom required along the path: the WBC does not model joint limits,
    #                    and a joint hitting its stop kicks the tip off the surface (seen: joint4 at 0.007 rad)

    def __init__(self, cfg, ctrl_model, qpos0, prior, f_des=5.0, passes=2, speed=0.03, weave=0.035, span=0.8,
                 inset=0.10, t0=0.0, scan_height=0.06):
        """prior: what the map knows about the station: (cx, cy, hx, hy, nominal top z) of the stand."""
        self.cfg, self.m, self.f_des, self.t0 = cfg, ctrl_model, f_des, t0
        self.cx, self.cy, self.hx, self.hy, self.z_nom = prior
        ix = Indexer(ctrl_model, cfg)
        self.ix, self.stow = ix, ix.arm_stow
        self.base_pos0, self.base_quat0 = qpos0[0:3].copy(), qpos0[3:7].copy()
        R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, self.base_quat0)
        yaw = np.arctan2(R0[3], R0[0])
        self.base_quat_level = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.level_err = np.zeros(3)
        mujoco.mju_subQuat(self.level_err, self.base_quat_level, self.base_quat0)
        self.base_brace = self.base_pos0 - np.array([0, 0, cfg.brace_dz])
        self.q_ik = qpos0.copy(); self.q_ik[0:3] = self.base_brace; self.q_ik[3:7] = self.base_quat_level
        self.passes, self.weave, self.span, self.inset = passes, weave, span, inset
        # path in the horizontal plane: along y over +-span*hy, weaving 1.5 periods in x, `inset` in from the near edge
        u = np.linspace(0, 1, 400)
        xy = self.path_xy(u)
        self.T_pass = max(4.0, np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) / speed)
        self.T_ph = dict(settle=0.5, brace=1.0, to_scan=2.5, scan=0.8, to_hover=2.0, hover=0.5, approach=3.0,
                         sweep=passes * self.T_pass, dwell=1.0, lift=1.0, stow=2.5, hold=1.0)
        t, self.phases = 0.0, []
        for p in PHASES:
            self.phases.append((t, p))
            t += self.T_ph[p]
        self.T = t
        self.start = {p: ts for ts, p in self.phases}
        # scan pose: tool straight down over the panel centre, scan_height above the nominal top
        self.q_scan = self._ik(np.array([self.cx, self.cy, self.z_nom + scan_height]), R_TOOL_DOWN, full=True)
        if self.q_scan is None:
            raise RuntimeError("scan pose over the panel is unreachable from this stance")
        self.model, self.q_hover, self.captured = None, None, {}
        self.t_contact, self.t_last, self.wait_s, self.max_wait = None, None, 0.0, 6.0

    # ------------------------------------------------------------------ geometry
    def path_xy(self, u):
        u = np.asarray(u, float)
        y = self.cy + self.span * self.hy * (2 * u - 1)
        x = self.cx - self.hx + self.inset + self.weave * np.sin(3 * np.pi * u)
        return np.stack([x, y], -1)

    def path_point(self, u):
        """Surface point, outward normal and path tangent at u in [0, 1] - from the *believed* surface model."""
        x, y = self.path_xy(u)
        p = np.array([x, y, float(self.model.height(x, y))])
        n = self.model.normal(x, y).reshape(3)
        a, b = self.path_xy(min(u + 1e-3, 1)), self.path_xy(max(u - 1e-3, 0))
        t = np.array([a[0] - b[0], a[1] - b[1], float(self.model.height(*a) - self.model.height(*b))])
        return p, n, t / np.linalg.norm(t)

    def _ik(self, p, R, seed=None, full=False):
        ix = self.ix
        rng = np.random.default_rng(0)
        seeds = ([seed] if seed is not None else []) + [np.array(s, float) for s in self.cfg.ik_seeds]
        seeds += [rng.uniform(0.9 * ix.arm_lo, 0.9 * ix.arm_hi) for _ in range(25)]
        span = ix.arm_hi - ix.arm_lo
        best, best_score = None, -np.inf
        for s in seeds:
            q, res = ik_arm(self.m, self.q_ik, ix, p, R, seed=s, iters=300 if full else 200, axis_only=not full)
            score = np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span)
            if res < 1e-3 and score > best_score:
                best, best_score = q, score
            if seed is not None and best is not None:
                break
        return best

    def set_model(self, model):
        """Install the reconstructed surface; plan the hover pose and check (and if needed shrink) the path's reach."""
        self.model = model
        # The arm cannot tilt the tool far *towards* the robot (the surface leaning away): pick the longest path
        # placement (length along the panel, weave, inset from the near edge) that is reachable at every point on
        # the reconstructed surface.
        span0, weave0, inset0 = self.span, self.weave, self.inset
        cands = [(sp, wv, ins) for sp in (span0, 0.85 * span0, 0.7 * span0, 0.55 * span0, 0.4 * span0)
                 for wv in (weave0, 0.6 * weave0, 0.3 * weave0) for ins in (inset0, inset0 - 0.025, inset0 - 0.05)]
        ix = self.ix
        margin = lambda qq: np.min(np.minimum(qq - ix.arm_lo, ix.arm_hi - qq))
        for min_margin, (self.span, self.weave, self.inset) in [(mm, c) for mm in (self.MIN_MARGIN, 0.5 * self.MIN_MARGIN)
                                                                  for c in cands]:
            q, ok = None, True
            for u in np.linspace(0, 1, 25):
                p, n, t = self.path_point(u)
                R = tool_frame(n, t)
                q_new = self._ik(p, R, seed=q)
                if q_new is not None and margin(q_new) < min_margin:
                    q_new = self._ik(p, R)  # multi-start: the solution furthest from the joint limits
                if q_new is None or margin(q_new) < min_margin:
                    ok = False
                    break
                q = q_new
            if ok:
                break
        else:
            raise RuntimeError("no path on the measured surface can be traced with the tool normal to it from this "
                               "stance (the surface leans too far away from the robot for the arm) - dock elsewhere")
        self.path_choice = dict(span_m=2 * self.span * self.hy, weave_m=self.weave, inset_m=self.inset,
                                min_joint_margin_rad=min_margin)
        p0, n0, t0 = self.path_point(0.0)
        self.q_hover = self._ik(p0 + self.HOVER * n0, tool_frame(n0, t0))
        if self.q_hover is None:
            raise RuntimeError("hover pose unreachable")

    # ------------------------------------------------------------------ timeline
    def phase(self, t):
        tr, name = t - self.t0, self.phases[0][1]
        for ts, p in self.phases:
            if tr >= ts:
                name = p
        return name

    def u(self, tr):
        if tr < self.start["sweep"]:
            return 0.0
        k, rem = divmod(min(tr - self.start["sweep"], self.T_ph["sweep"] - 1e-9), self.T_pass)
        s = cruise(rem, self.T_pass, min(1.0, 0.15 * self.T_pass))
        return s if int(k) % 2 == 0 else 1.0 - s

    def _joint_move(self, t, q_from, q_to, phase):
        f = lambda tt: q_from + min_jerk((tt - self.t0 - self.start[phase]) / self.T_ph[phase]) * (q_to - q_from)
        return fd(f, t)

    def ref(self, t, tip_pos, arm_q, f_meas, f_vec=None):
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        if self.phase(t) == "approach":  # contact-triggered: keep approaching (clock paused) until touched
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
        if ph in ("settle", "brace", "hold"):
            return r
        if ph == "to_scan":
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = self._joint_move(t, self.stow, self.q_scan, "to_scan")
            return r
        if ph == "scan":
            r["arm_q"] = self.q_scan
            return r
        if self.model is None:
            raise RuntimeError("set_model() must be called at the end of the scan phase")
        if ph == "to_hover":
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = self._joint_move(t, self.q_scan, self.q_hover, "to_hover")
            return r
        if ph == "stow":
            q_s = self.captured.setdefault("stow_q", arm_q.copy())
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = self._joint_move(t, q_s, self.stow, "stow")
            return r

        u = self.u(tr)
        p_s, n, tan = self.path_point(u)
        task = dict(w_ee=30.0, w_post=1e-4, arm_q=self.q_hover, ee_quat=mat2quat(tool_frame(n, tan)), axis_only=True)
        if ph == "hover":
            task.update(ee_pos=p_s + self.HOVER * n, ee_vel=np.zeros(3), ee_acc=np.zeros(3))
        elif ph == "lift":
            p_l = self.captured.setdefault("lift_p", tip_pos.copy())
            f = lambda tt: p_l + min_jerk((tt - self.t0 - self.start["lift"]) / self.T_ph["lift"]) * 0.05 * n
            task["ee_pos"], task["ee_vel"], task["ee_acc"] = fd(f, t)
        else:  # approach / sweep / dwell: force control along the surface normal
            p, v, a = fd(lambda tt: self.path_point(self.u(tt - self.t0))[0], t)
            _, ndot, _ = fd(lambda tt: self.path_point(self.u(tt - self.t0))[1], t)
            task.update(ee_pos=p, ee_vel=v, ee_acc=a, ee_omega=np.cross(n, ndot),   # tool axis = -n turns at n x n'
                        force=dict(n=n, f_des=self.f_des, f_meas=f_meas))
        r.update(task)
        return r
