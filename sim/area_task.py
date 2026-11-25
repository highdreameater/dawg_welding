"""Area coverage with the arm while the stance WBC holds the base: boustrophedon (raster) strokes over a *measured*
surface patch. One generic surface type - no wall / pipe / panel cases.

    SurfacePatch : the measured surface w(u, v) (sim/perception/work_area.py) over u in [u0, u1], v in [0, H].
                   point(a, b) / tangent(a, b) with a = u - u0, b = v; the tool axis is the fitted normal (into it).

Split of work:
    quadruped : the stance WBC (sim/wbc/stance_wbc.py) only stabilises the robot - base pose, feet, friction cones -
                and tracks the arm joints as a posture reference
    arm       : admittance on the measured contact force along the fitted normal (5 N) + damped-least-squares IK
                (AreaTask.ref). Model errors along the normal are absorbed by the admittance; the desired contact force
                is fed to the WBC dynamics so the joints push by torque instead of yielding; a tangential integral loop
                on the measured tip (forward kinematics) removes what slide remains.

Reachability decides the stance and the patches (no per-shape tuning):
    plan_stance(...)    : IK over the measured surface from virtual braced stances: the standoff D and a greedy
                          tiling of the area into patches, each reachable on every raster row with joint margin from
                          its own stance, also at D +- 3 cm, and ending 3 cm short of the reach edge (dock error)
    reachable_span(...) : the same check from the robot's *actual* stance before tracing; what is out of reach is
                          handed back as a new patch

Phases: settle -> brace -> unstow (joint space to the first hover pose) -> hover -> approach -> raster -> lift ->
stow -> hold. The raster is a sequence of segments, each timed by its arc length on the measured surface.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from force_task import cruise, fd, min_jerk
from wbc.stance_wbc import Indexer, ik_arm, initial_state

PHASES = ["settle", "brace", "unstow", "hover", "approach", "raster", "lift", "stow", "hold"]
# WBC arm-posture gains while tracing. The posture task is in acceleration space, so its stiffness scales with the
# inertia a joint moves: at the default kp 400 the Z1 wrist (~0.013 kg m^2) is ~5 Nm/rad and its 1 Nm of joint
# friction tilted the tool ~5 deg on the waves. kp 2500 / kd 100 (~8 Hz) -> 1 deg tracking, 1 mm tangential error.
TRACE_WBC_GAINS = dict(kp_post=2500.0, kd_post=100.0)
REACH_MARGIN = 0.03          # min joint-limit margin (fraction of range, ~5-10 deg on the Z1) to count as reachable


def tool_rotation(x, t):
    """x = tool axis, t = a unit vector perpendicular to it (surface tangent). Roll about x is free anyway."""
    x = x / np.linalg.norm(x)
    t = t - (t @ x) * x
    t /= np.linalg.norm(t)
    return np.stack([x, np.cross(t, x), t], 1)


# ----------------------------------------------------------------------------------------------- measured surface
class SurfacePatch:
    kind = "surface"

    def __init__(self, frame, surface, u0, u1):
        """frame: AreaFrame (world placement), surface: MeasuredSurface (area coordinates), [u0, u1]: the u-range."""
        self.f, self.s = frame, surface
        self.u0, self.u1 = float(u0), float(u1)
        self.width, self.height = self.u1 - self.u0, surface.H

    def point(self, a, b):
        u, v = self.u0 + a, b
        p = self.f.to_world(u, v, float(self.s.w(u, v)))
        nu = self.s.normal_uvw(u, v)
        n = nu[0] * self.f.ex + nu[1] * self.f.ey + nu[2] * self.f.ez
        return p, n / np.linalg.norm(n)

    def tangent(self, a, b, h=1e-3):
        t = self.point(a + h, b)[0] - self.point(a - h, b)[0]
        return t / np.linalg.norm(t)

    def arc_length(self, ab0, ab1, n=24):
        P = [self.point(*((1 - s) * np.asarray(ab0) + s * np.asarray(ab1)))[0] for s in np.linspace(0, 1, n)]
        return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

    def sub(self, u0, u1):
        return SurfacePatch(self.f, self.s, u0, u1)

    def mapped(self, f_point, f_vec):
        return SurfacePatch(self.f.mapped(f_point, f_vec), self.s, self.u0, self.u1)


def target_pose(area, ab, tilt=0.0):
    """(tool tip point on the surface, tool rotation, surface normal) at surface coordinates ab."""
    p, n = area.point(*ab)
    x = -n if tilt == 0.0 else np.cos(tilt) * (-n) - np.sin(tilt) * np.array([0, 0, 1.0])
    return p, tool_rotation(x, area.tangent(*ab)), n


# ----------------------------------------------------------------------------------------------- reachability
def braced_qpos(cfg, qpos):
    """qpos with the base lowered by brace_dz and levelled (the pose the arm works from)."""
    q = qpos.copy()
    q[2] -= cfg.brace_dz
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q[3:7])
    yaw = np.arctan2(R[3], R[0])
    q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    return q


class Reach:
    """IK reachability of a measured surface from a given (braced) base pose."""

    def __init__(self, cfg, ctrl_model):
        self.cfg, self.m = cfg, ctrl_model
        self.ix = Indexer(ctrl_model, cfg)
        self.span = self.ix.arm_hi - self.ix.arm_lo
        self.d = mujoco.MjData(ctrl_model)

    def _solve(self, q_base, p, R, seed):
        q, res = ik_arm(self.m, q_base, self.ix, p, R, seed=seed, iters=150, axis_only=True)
        d = self.d
        d.qpos[:] = q_base
        d.qpos[self.ix.arm_q] = q
        mujoco.mj_kinematics(self.m, d)
        ax = np.degrees(np.arccos(np.clip(d.site_xmat[self.ix.tip].reshape(3, 3)[:, 0] @ R[:, 0], -1, 1)))
        mg = float(np.min(np.minimum(q - self.ix.arm_lo, self.ix.arm_hi - q) / self.span))
        return q, (mg if (res < 2e-3 and ax < 2.0) else -1.0)

    def _cold(self, q_base, p, R):
        return max((self._solve(q_base, p, R, np.array(s, float)) for s in self.cfg.ik_seeds), key=lambda t: t[1])

    def margins(self, q_base, area, us, vs):
        """Joint margin (min over the rows vs) at each u of `us` (area u coordinates, a = u - area.u0);
        -1 = unreachable. Each row is swept outwards from the u closest to the base, warm-started."""
        mid = vs[len(vs) // 2]
        c = int(np.argmin([np.linalg.norm(area.point(u - area.u0, mid)[0][:2] - q_base[:2]) for u in us]))
        out = np.full((len(vs), len(us)), -1.0)
        for r, v in enumerate(vs):
            p, R, _ = target_pose(area, (us[c] - area.u0, v))
            q_c, out[r, c] = self._cold(q_base, p, R)
            for order in (range(c + 1, len(us)), range(c - 1, -1, -1)):
                q = q_c
                for i in order:
                    p, R, _ = target_pose(area, (us[i] - area.u0, v))
                    q, mg = self._solve(q_base, p, R, q)
                    if mg < 0:                         # retry cold before calling it unreachable
                        q2, mg2 = self._cold(q_base, p, R)
                        if mg2 > mg:
                            q, mg = q2, mg2
                    out[r, i] = mg
        return out.min(0)


def contiguous_span(us, ok, around):
    """(u_first, u_last) of the run of True in ok that contains index `around` (else the longest run), or None."""
    runs, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return None
    hit = [r for r in runs if r[0] <= around <= r[1]]
    i, j = hit[0] if hit else max(runs, key=lambda r: r[1] - r[0])
    return float(us[i]), float(us[j])


def virtual_stance(cfg, ctrl_model, xy, yaw, ground_z):
    """Nominal stance (arm stowed) at a base pose standing on ground at height ground_z, braced."""
    dv = mujoco.MjData(ctrl_model)
    initial_state(ctrl_model, dv, Indexer(ctrl_model, cfg), xy=xy, yaw=yaw)
    dv.qpos[2] += ground_z
    return braced_qpos(cfg, dv.qpos.copy())


@dataclass
class StancePlan:
    D: float                    # standoff: base centre -> median surface depth, along the area normal
    w_ref: float                # that median surface depth (area w)
    rel: tuple                  # reachable u-interval relative to the base's lateral position (robust to dock error)
    patches: list               # [(u0, u1, s)]: patch u-range and the base's lateral position for it
    table: list                 # per D: (D, robust reachable width from the centre stance)
    gaps: list = None           # u-ranges no stance could reach


def plan_stance(cfg, ctrl_model, frame, surface, ground, D_grid=None, du=0.03, lat_tol=0.03, d_tol=0.03, rows=3,
                n_candidates=3):
    """Choose the standoff and split the area into patches from IK reachability on the measured surface.

    1. reach width per standoff D from a stance facing the area centre; robust = intersected with D +- d_tol (3 cm:
       the marker-corrected dock; the actual stance is re-checked before tracing anyway)
    2. for the best few D: greedy tiling from the left edge. For each patch start a, candidate stances around
       'a sits just inside the left reach edge' are checked by IK over the measured surface: with joint margin at D,
       merely feasible at D +- d_tol;
       the stance whose reachable run from a (- lat_tol) extends furthest wins; the patch ends lat_tol short of the
       run's end (dock-error margin). Non-uniform surfaces (waves, curvature) get patches sized by their own reach.
    3. D = fewest patches (then the widest centre reach).
    ground: the ground height under the robot, or ground(xy, yaw) -> height of the map's ground for a stance there,
    None if it is not a valid stance (e.g. feet straddling a step) - a stair platform is a valid, higher stance."""
    reach = Reach(cfg, ctrl_model)
    D_grid = np.round(np.arange(0.55, 0.951, d_tol), 3) if D_grid is None else D_grid    # neighbours = +-d_tol
    W = frame.W
    area = SurfacePatch(frame, surface, 0.0, W)
    vs = np.linspace(surface.H, 0.0, rows)
    w_ref = float(np.median(surface.w(*np.meshgrid(np.linspace(0, W, 9), vs))))
    yaw = float(np.arctan2(-frame.ez[1], -frame.ez[0]))
    ground_fn = ground if callable(ground) else (lambda xy, yaw_: ground)

    def q_at(D, s):                                   # virtual braced stance, standing on the map's ground there
        xy = (frame.o + s * frame.ex + (w_ref + D) * frame.ez)[:2]
        g = ground_fn(xy, yaw)
        return None if g is None else virtual_stance(cfg, ctrl_model, xy, yaw, g)

    def margins(D, s, us_):
        q = q_at(D, s)
        return np.full(len(us_), -1.0) if q is None else reach.margins(q, area, us_, vs)
    s_c = W / 2
    us = np.unique(np.r_[np.arange(0.0, W, du), W])
    ivs = []                                          # per D: (span with joint margin, span merely feasible)
    ic = int(np.argmin(np.abs(us - s_c)))
    for D in D_grid:
        mg = margins(D, s_c, us)
        sp = [contiguous_span(us, mg >= thr, ic) for thr in (REACH_MARGIN, 0.0)]
        ivs.append([None if x is None else (x[0] - s_c, x[1] - s_c) for x in sp])
    table, cands = [], []
    for k, D in enumerate(D_grid):
        # nominal stance: with joint margin; the +-d_tol (dock error) stances: only feasible
        nb = [ivs[j][0 if j == k else 1] for j in (k - 1, k, k + 1) if 0 <= j < len(ivs)]
        if len(nb) < 3 or any(iv is None for iv in nb):
            table.append((float(D), 0.0))
            continue
        lo, hi = max(iv[0] for iv in nb), min(iv[1] for iv in nb)
        table.append((float(D), round(max(0.0, hi - lo), 3)))
        if hi - lo >= 0.06:
            cands.append((hi - lo, float(D), (float(lo), float(hi))))
    if not cands:
        raise RuntimeError(f"work area not reachable from any standoff (robust reach widths {table})")
    cands.sort(reverse=True)
    best = None
    for wdt, D, (lo, hi) in cands[:n_candidates]:
        patches, gaps = _tile(margins, D, d_tol, lo, hi, W, du, lat_tol)
        key = (len(gaps) > 0, len(patches), -wdt)
        if best is None or key < best[0]:
            best = (key, D, (lo, hi), patches, gaps)
        if len(patches) == 1 and not gaps:
            break
    _, D, rel, patches, gaps = best
    if not patches:
        raise RuntimeError(f"no reachable patch found (robust reach widths {table})")
    return StancePlan(D, w_ref, rel, patches, table, gaps)


def _tile(margins, D, d_tol, lo, hi, W, du, lat_tol):
    patches, gaps, a = [], [], 0.0
    while a < W - 1e-6:
        u_lo = max(0.0, a - lat_tol)
        us = np.unique(np.r_[np.arange(u_lo, min(W, a + (hi - lo) + 0.1), du), min(W, a + (hi - lo) + 0.1)])
        s0 = a - lo - lat_tol                         # a sits lat_tol inside the nominal left reach edge
        best = None
        for off in (0.0, 0.05, -0.05, 0.10, -0.10):
            s = s0 + off
            ok = np.ones(len(us), bool)
            for DD in (D, D - d_tol, D + d_tol):       # nominal: joint margin; dock-error stances: feasible
                ok &= margins(DD, s, us) >= (REACH_MARGIN if DD == D else 0.0)
                if not ok[0]:
                    break
            if not ok[0]:
                continue
            j = int(np.argmin(ok)) - 1 if not ok.all() else len(us) - 1
            b = float(us[j])
            if best is None or b > best[0]:
                best = (b, s)
            if b >= W - 1e-6:
                break
        if best is None:
            gaps.append((a, min(W, a + du)))
            a += du
            continue
        b, s = best
        end = W if b >= W - 1e-6 else b - lat_tol
        if end - a < 0.03:
            gaps.append((a, min(W, a + du)))
            a += du
            continue
        patches.append((float(a), float(end), float(s)))
        a = end
    return patches, gaps


def stance_pose(frame, w_ref, D, s):
    xy = (frame.o + s * frame.ex + (w_ref + D) * frame.ez)[:2]
    return np.array([*xy, np.arctan2(-frame.ez[1], -frame.ez[0])])


def reachable_span(cfg, ctrl_model, qpos, patch, du=0.02, rows=3):
    """From the actual (to be braced) stance: the largest reachable u sub-range (area u) of `patch`, or None."""
    reach = Reach(cfg, ctrl_model)
    us = np.linspace(patch.u0, patch.u1, max(3, int(np.ceil(patch.width / du)) + 1))
    vs = np.linspace(patch.height, 0.0, rows)
    mg = reach.margins(braced_qpos(cfg, qpos), patch, us, vs)
    return contiguous_span(us, mg >= REACH_MARGIN, len(us) // 2), mg


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

    def __init__(self, cfg, ctrl_model, qpos0, area, t0=0.0, speed=0.05, spacing=0.02, f_des=5.0, k_adm=0.004,
                 tilt=0.0):
        """area: SurfacePatch (the robot's *measured* geometry, in the same frame as qpos0). spacing: stroke
        spacing (the brush is 2.2 cm wide: 2 cm strokes overlap slightly, so the patch reads as filled).
        tilt: tool tilt below the normal [rad]; 0 = tool exactly along the fitted normal."""
        self.cfg, self.area, self.t0, self.f_des, self.tilt = cfg, area, t0, f_des, tilt
        self.speed, self.spacing = speed, spacing
        ix = Indexer(ctrl_model, cfg)
        self.ix, self.stow = ix, ix.arm_stow
        self.base_pos0, self.base_quat0 = qpos0[0:3].copy(), qpos0[3:7].copy()
        qb = braced_qpos(cfg, qpos0)
        self.base_quat_level = qb[3:7].copy()
        self.level_err = np.zeros(3)
        mujoco.mju_subQuat(self.level_err, self.base_quat_level, self.base_quat0)
        self.base_brace = qb[0:3].copy()

        self.segs = raster_segments(area.width, area.height, spacing)
        self.seg_T, self.seg_t0 = [], []
        t = 0.0
        for kind, p0, p1 in self.segs:
            L = area.arc_length(p0, p1)
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
        p0, R0_, n0 = self.target(self.segs[0][1])
        rng = np.random.default_rng(0)
        seeds = [np.array(s, float) for s in cfg.ik_seeds] + [rng.uniform(0.9 * ix.arm_lo, 0.9 * ix.arm_hi) for _ in range(30)]
        span = ix.arm_hi - ix.arm_lo
        best, best_m = None, -np.inf
        for s in seeds:
            q, res = ik_arm(ctrl_model, qb, ix, p0 + self.HOVER * n0, R0_, seed=s, iters=200, axis_only=True)
            mg = np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span)
            if res < 1e-3 and mg > best_m:
                best, best_m = q, mg
        if best is None:
            raise RuntimeError("first hover pose unreachable from this stance")
        self.q_hover = best
        q, worst, margin = best, 0.0, best_m
        for _, a, b in self.segs:
            for ab in (a, b):
                p, R, _ = self.target(ab)
                q, res = ik_arm(ctrl_model, qb, ix, p, R, seed=q, axis_only=True)
                if res > 2e-3:                       # warm start stuck: retry cold (as the reach planner does)
                    for s in [best] + [np.array(x, float) for x in cfg.ik_seeds]:
                        q2, r2 = ik_arm(ctrl_model, qb, ix, p, R, seed=s, axis_only=True)
                        if r2 < res:
                            q, res = q2, r2
                worst = max(worst, res)
                margin = min(margin, np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span))
        self.ik_residual_mm, self.ik_margin = 1000 * worst, float(margin)
        if worst > 2e-3:
            raise RuntimeError(f"raster not reachable from this stance (IK residual {1000 * worst:.1f} mm)")
        self.captured = {}
        self.t_contact, self.t_last, self.wait_s, self.max_wait = None, None, 0.0, 6.0
        # arm admittance state + IK workspace (controller model, own MjData)
        self.m, self.d = ctrl_model, mujoco.MjData(ctrl_model)
        self.k_adm = k_adm
        self.delta, self.q_cmd, self.qd_f = -self.HOVER, self.stow.copy(), np.zeros(len(self.stow))
        # tangential outer loop on the *measured* tip (forward kinematics of the measured joints): the contact force
        # is not in the WBC's model, it deflects the joint position control and on a frictionless contact that
        # deflection slides the tip along the surface (5-6 cm down on a wall). Integral correction, tangent plane only
        # (the normal direction belongs to the force admittance).
        self.k_tan, self.tan_max, self.t_corr = 4.0, 0.10, np.zeros(3)
        self.tip_err = np.zeros(3)
        # expected contact force fed to the WBC dynamics (ref["f_ext"] = f_ff * n), ramped in over ff_ramp s once in
        # contact and out when contact is lost / on lift: the joints push by torque instead of yielding
        self.f_ff, self.ff_ramp = 0.0, 0.2

    def target(self, ab):
        return target_pose(self.area, ab, self.tilt)

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

        # hover / approach / raster / lift: admittance offset along the normal, tangential outer loop, then IK
        if ph == "hover":
            self.delta = -self.HOVER
            self.t_corr[:] = 0.0
        elif ph in ("approach", "raster"):
            self.delta = float(np.clip(self.delta + self.k_adm * (self.f_des - f_meas) * dt, -self.HOVER, 0.03))
        else:  # lift: back off 6 cm
            self.delta = max(self.delta - 0.06 * dt / self.T_ph["lift"], -0.06)
        p, R, n = self.target(self.ab(tr))
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        e = p - self.d.site_xpos[self.ix.tip]
        self.tip_err = e - (e @ n) * n                       # measured tangential tip error
        if ph in ("approach", "raster"):
            self.t_corr += self.k_tan * self.tip_err * dt
        self.t_corr -= (self.t_corr @ n) * n                 # stay in the current tangent plane
        nc = np.linalg.norm(self.t_corr)
        if nc > self.tan_max:
            self.t_corr *= self.tan_max / nc
        q_prev = self.q_cmd.copy()
        self._ik_step(qpos, p - self.delta * n + self.t_corr, R)
        want = self.f_des if (ph in ("approach", "raster") and f_meas > 1.0) else 0.0
        step = self.f_des * dt / self.ff_ramp
        self.f_ff += float(np.clip(want - self.f_ff, -step, step))
        r["f_ext"] = self.f_ff * n
        qd = (self.q_cmd - q_prev) / max(dt, 1e-6) if dt > 0 else np.zeros(n_arm)
        self.qd_f += 0.2 * (qd - self.qd_f)          # light filter on the joint velocity feed-forward
        r.update(arm_q=self.q_cmd.copy(), arm_qd=self.qd_f.copy())
        return r
