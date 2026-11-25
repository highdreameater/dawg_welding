"""Area coverage with the arm while the stance WBC holds the base: boustrophedon (raster) strokes over a measured
surface patch, not just one line.

    WallArea : a rectangle on a measured wall plane
    PipeArea : a band of a measured vertical pipe (arc +-theta_max around the robot-facing direction, z band)
Both are traced in contact at 5 N along the measured surface normal, staying in contact on the step-downs between
strokes. Split of work: the stance WBC (sim/wbc/stance_wbc.py) only stabilises the quadruped (base pose, feet,
friction cones) and tracks the arm joints as a posture reference; the arm runs a basic admittance loop on the
measured contact force + damped-least-squares IK (AreaTask.ref). The force-shaded trail
(sim/paint.py) shows the work: lighter = less force, darker = more.

Tool angle: the 27 cm torch is held 30 deg below the surface normal (push angle). With the tool exactly normal
to a vertical surface the Z1 on a Go2 has almost no workspace at these heights (it cannot fold the elbow enough
for targets 0.7-0.9 m ahead); at 30 deg the joint-limit margin over the whole area is 15-25 % of the range.

Phases: settle -> brace -> unstow (joint space to the first hover pose) -> hover -> approach -> raster -> lift ->
stow -> hold. The raster is a sequence of segments, each timed with a cruise profile (constant speed, smooth ramps).
"""
import mujoco
import numpy as np

from force_task import cruise, fd, min_jerk
from wbc.stance_wbc import Indexer, ik_arm

PHASES = ["settle", "brace", "unstow", "hover", "approach", "raster", "lift", "stow", "hold"]
TILT = np.radians(30.0)


def tool_rotation(x, t):
    """x = tool axis, t = a unit vector perpendicular to it (horizontal surface tangent)."""
    x = x / np.linalg.norm(x)
    t = t - (t @ x) * x
    t /= np.linalg.norm(t)
    return np.stack([x, np.cross(t, x), t], 1)


# ----------------------------------------------------------------------------------------------- surfaces (belief)
class WallArea:
    """Measured wall plane + the target rectangle on it. Coordinates (a, b): a along u (right, seen from the robot),
    b along v (up), metres from the rectangle's lower-left corner."""
    kind = "paint"

    def __init__(self, p0, n, u, width, height):
        self.p0, self.n, self.u = (np.asarray(x, float) for x in (p0, n, u))
        self.v = np.cross(self.n, self.u)            # n x u = up for u = right (n towards the robot)
        self.width, self.height = width, height

    def point(self, a, b):
        return self.p0 + a * self.u + b * self.v, self.n

    def tangent(self, a, b):
        return self.u


class PipeArea:
    """Measured vertical pipe: axis (x, y), radius, robot-facing reference direction e1; a = arc length around the
    pipe from -theta_max, b = height from z_lo."""
    kind = "pipe"

    def __init__(self, axis_xy, radius, e1, theta_max, z_lo, z_hi):
        self.c = np.asarray(axis_xy, float)
        self.r, self.th_max = radius, theta_max
        self.e1 = np.array([*e1[:2], 0.0]) / np.linalg.norm(e1[:2])
        self.e2 = np.cross([0, 0, 1.0], self.e1)
        self.z_lo = z_lo
        self.width, self.height = 2 * theta_max * radius, z_hi - z_lo

    def theta(self, a):
        return -self.th_max + a / self.r

    def point(self, a, b):
        th = self.theta(a)
        n = np.cos(th) * self.e1 + np.sin(th) * self.e2
        return np.array([*self.c, self.z_lo + b]) + self.r * n, n

    def tangent(self, a, b):
        th = self.theta(a)
        return -np.sin(th) * self.e1 + np.cos(th) * self.e2


def tool_axis(n, up=np.array([0, 0, 1.0])):
    """Into the surface, tilted TILT downwards."""
    return np.cos(TILT) * (-n) - np.sin(TILT) * up


# ----------------------------------------------------------------------------------------------- raster plan
def raster_segments(width, height, spacing, inset=0.0):
    """Boustrophedon over [0, width] x [0, height], top row first. Returns [(kind, (a0, b0), (a1, b1))]."""
    rows = max(2, int(np.ceil((height - 2 * inset) / spacing)) + 1)
    bs = np.linspace(height - inset, inset, rows)
    segs, left = [], True
    for k, b in enumerate(bs):
        a0, a1 = (0.0, width) if left else (width, 0.0)
        segs.append(("stroke", (a0, b), (a1, b)))
        if k < rows - 1:
            segs.append(("step", (a1, b), (a1, bs[k + 1])))
        left = not left
    return segs


class AreaTask:
    HOVER = 0.04

    def __init__(self, cfg, ctrl_model, qpos0, area, t0=0.0, speed=0.05, spacing=0.02, f_des=5.0, k_adm=0.004):
        """area: WallArea / PipeArea (the robot's *measured* geometry). spacing: stroke spacing (the trail is
        2.2 cm wide: 2 cm strokes overlap slightly, so the patch reads as filled)."""
        self.cfg, self.area, self.t0, self.f_des = cfg, area, t0, f_des
        self.contact = True
        self.speed, self.spacing = speed, spacing
        ix = Indexer(ctrl_model, cfg)
        self.ix, self.stow = ix, ix.arm_stow
        self.base_pos0, self.base_quat0 = qpos0[0:3].copy(), qpos0[3:7].copy()
        R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, self.base_quat0)
        yaw = np.arctan2(R0[3], R0[0])
        self.base_quat_level = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.level_err = np.zeros(3)
        mujoco.mju_subQuat(self.level_err, self.base_quat_level, self.base_quat0)
        self.base_brace = self.base_pos0 - np.array([0, 0, cfg.brace_dz])

        self.segs = raster_segments(area.width, area.height, spacing)
        self.seg_T, self.seg_t0 = [], []
        t = 0.0
        for kind, p0, p1 in self.segs:
            L = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            T = max(1.0, L / speed + 0.4) if kind == "stroke" else max(0.8, 2.0 * L / speed)
            self.seg_t0.append(t); self.seg_T.append(T)
            t += T
        T_raster = t
        self.T_ph = dict(settle=0.5, brace=1.0, unstow=2.5, hover=0.6, approach=2.5,
                         raster=T_raster, lift=1.0, stow=2.5, hold=0.6)
        t, self.phases = 0.0, []
        for p in PHASES:
            self.phases.append((t, p)); t += self.T_ph[p]
        self.T = t
        self.start = {p: ts for ts, p in self.phases}

        # hover pose IK (multi-start) from the braced stance, then check the whole raster for reachability
        q_ik = qpos0.copy(); q_ik[0:3] = self.base_brace; q_ik[3:7] = self.base_quat_level
        p0, R0_, _ = self.target(self.segs[0][1])
        rng = np.random.default_rng(0)
        seeds = [np.array(s, float) for s in cfg.ik_seeds] + [rng.uniform(0.9 * ix.arm_lo, 0.9 * ix.arm_hi) for _ in range(30)]
        span = ix.arm_hi - ix.arm_lo
        best, best_m = None, -np.inf
        n0 = self.area.point(*self.segs[0][1])[1]
        for s in seeds:
            q, res = ik_arm(ctrl_model, q_ik, ix, p0 + self.HOVER * n0, R0_, seed=s, iters=200, axis_only=True)
            mg = np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span)
            if res < 1e-3 and mg > best_m:
                best, best_m = q, mg
        if best is None:
            raise RuntimeError(f"{area.kind}: first hover pose unreachable from this stance")
        self.q_hover = best
        q, worst, margin = best, 0.0, best_m
        for _, a, b in self.segs:
            for ab in (a, b):
                p, R, _ = self.target(ab)
                q, res = ik_arm(ctrl_model, q_ik, ix, p, R, seed=q, axis_only=True)
                worst = max(worst, res)
                margin = min(margin, np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span))
        self.ik_residual_mm, self.ik_margin = 1000 * worst, float(margin)
        if worst > 2e-3:
            raise RuntimeError(f"{area.kind}: raster not reachable from this stance (IK residual {1000 * worst:.1f} mm)")
        self.captured = {}
        self.t_contact, self.t_last, self.wait_s, self.max_wait = None, None, 0.0, 6.0
        # arm admittance state + IK workspace (controller model, own MjData)
        self.m, self.d = ctrl_model, mujoco.MjData(ctrl_model)
        self.k_adm = k_adm
        self.delta, self.q_cmd, self.qd_f = -self.HOVER, self.stow.copy(), np.zeros(len(self.stow))

    # ------------------------------------------------------------------ geometry
    def target(self, ab):
        """(tool tip point on the surface, tool rotation, surface normal) at surface coordinates ab."""
        p, n = self.area.point(*ab)
        x = tool_axis(n)
        R = tool_rotation(x, self.area.tangent(*ab))
        return p, R, n

    # ------------------------------------------------------------------ timeline
    def phase(self, t):
        tr, name = t - self.t0, self.phases[0][1]
        for ts, p in self.phases:
            if tr >= ts:
                name = p
        return name

    def seg_at(self, tr):
        """(segment index, progress 0..1) at relative time tr (clamped to the raster)."""
        tau = np.clip(tr - self.start["raster"], 0.0, self.T_ph["raster"] - 1e-9)
        k = int(np.searchsorted(self.seg_t0, tau, side="right") - 1)
        T = self.seg_T[k]
        rem = tau - self.seg_t0[k]
        s = cruise(rem, T, min(0.4, 0.3 * T)) if self.segs[k][0] == "stroke" else min_jerk(rem / T)
        return k, s

    def ab(self, tr):
        if tr < self.start["raster"]:
            return np.array(self.segs[0][1], float)
        k, s = self.seg_at(tr)
        _, a, b = self.segs[k]
        return (1 - s) * np.array(a, float) + s * np.array(b, float)

    def progress(self, t):
        if self.phase(t) != "raster":
            return 0.0 if t - self.t0 < self.start["raster"] else 1.0
        return (t - self.t0 - self.start["raster"]) / self.T_ph["raster"]

    # ------------------------------------------------------------------ arm: admittance + differential IK
    def _ik_step(self, qpos, p_des, R_des, iters=2):
        """Damped-least-squares IK from the current arm command (base at its measured pose): position + tool axis."""
        m, d, ix = self.m, self.d, self.ix
        d.qpos[:] = qpos
        d.qpos[ix.arm_q] = self.q_cmd
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cols = ix.arm_v[:6]
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            Rc = d.site_xmat[ix.tip].reshape(3, 3)
            e = np.concatenate([p_des - d.site_xpos[ix.tip], 0.5 * np.cross(Rc[:, 0], R_des[:, 0])])
            mujoco.mj_jacSite(m, d, jp, jr, ix.tip)
            jr_ = (np.eye(3) - np.outer(Rc[:, 0], Rc[:, 0])) @ jr      # roll about the tool axis is free
            J = np.vstack([jp[:, cols], 0.5 * jr_[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[ix.arm_q[:6]] = np.clip(d.qpos[ix.arm_q[:6]] + dq, ix.arm_lo[:6], ix.arm_hi[:6])
        self.q_cmd = d.qpos[ix.arm_q].copy()

    def ref(self, t, qpos, f_meas):
        """References for the stance WBC. The WBC only stabilises the robot: base pose + feet, and it tracks the arm
        joints as a posture reference. The arm itself runs a plain admittance loop:
            tool point = path point on the measured surface - delta * n      (delta: how far 'into' the surface)
            d(delta)/dt = k_a (F_des - F_meas)  in contact phases   (no contact -> creeps in at k_a F_des = 2 cm/s)
        and damped-least-squares IK turns the tool pose into the arm joint targets."""
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        ph = self.phase(t)
        if ph == "approach":   # contact-triggered start of the raster: pause the clock until 0.5 s after touch-down
            if f_meas > 1.0:
                self.t_contact = self.t_contact or t
            if (t - self.t0 > self.start["raster"] - dt and (self.t_contact is None or t - self.t_contact < 0.5)
                    and self.wait_s < self.max_wait):
                self.t0 += dt
                self.wait_s += dt
        tr = t - self.t0
        n_arm = len(self.stow)
        s_brace = min_jerk((tr - self.start["brace"]) / self.T_ph["brace"])
        qb = self.base_quat0.copy()
        mujoco.mju_quatIntegrate(qb, self.level_err, s_brace)
        r = dict(base_pos=self.base_pos0 + s_brace * (self.base_brace - self.base_pos0), base_quat=qb,
                 arm_q=self.stow, arm_qd=np.zeros(n_arm), arm_qdd=np.zeros(n_arm), w_ee=0.0, w_post=10.0)
        if ph in ("settle", "brace", "hold"):
            return r
        if ph == "unstow":
            f = lambda tt: self.stow + min_jerk((tt - self.t0 - self.start["unstow"]) / self.T_ph["unstow"]) * (self.q_hover - self.stow)
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = fd(f, t)
            self.q_cmd = r["arm_q"].copy()
            return r
        if ph == "stow":
            q_s = self.captured.setdefault("stow_q", self.q_cmd.copy())
            f = lambda tt: q_s + min_jerk((tt - self.t0 - self.start["stow"]) / self.T_ph["stow"]) * (self.stow - q_s)
            r["arm_q"], r["arm_qd"], r["arm_qdd"] = fd(f, t)
            return r

        # hover / approach / raster / lift: admittance offset along the normal, then IK
        if ph == "hover":
            self.delta = -self.HOVER
        elif ph in ("approach", "raster"):
            self.delta = float(np.clip(self.delta + self.k_adm * (self.f_des - f_meas) * dt, -self.HOVER, 0.03))
        else:  # lift: back off 6 cm
            self.delta = max(self.delta - 0.06 * dt / self.T_ph["lift"], -0.06)
        p, R, n = self.target(self.ab(tr))
        q_prev = self.q_cmd.copy()
        self._ik_step(qpos, p - self.delta * n, R)
        qd = (self.q_cmd - q_prev) / max(dt, 1e-6) if dt > 0 else np.zeros(n_arm)
        self.qd_f += 0.2 * (qd - self.qd_f)          # light filter on the joint velocity feed-forward
        r.update(arm_q=self.q_cmd.copy(), arm_qd=self.qd_f.copy())
        return r
