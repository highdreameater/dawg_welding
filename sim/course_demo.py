"""Full mission: RL walk -> climb stairs -> stop -> hand over to the stance WBC -> weld a crack in a wall on the left.

Course (world frame, robot starts at x=0 facing +x):
    flat 3 m -> 6 steps x 10 cm (30 cm run) -> 3 m platform (top 0.6 m); a wall runs along the platform's left edge
    (inner face at y = +0.58, 1 m tall) with a figurative zig-zag crack 0.52-0.70 m above the platform.

Controllers:
    RL   : pretrained Go2 policy (MoE-CTS or symmetry), 50 Hz, PD on legs; Z1 held folded (PD + gravity comp)
    blend: 0.5 s linear torque blend RL -> WBC (RL leg targets frozen at hand-over)
    WBC  : sim/wbc/stance_wbc.py (go2_z1 config) - brace, unfold arm, approach, weld along the crack, retract, stow

    python sim/course_demo.py                     # both policies -> results/course/
    python sim/course_demo.py --policy symmetry
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from locomotion_demo import POLICIES  # noqa: E402  (policy interfaces: cmd scale, gains, action scale)
from wbc.robots import ROOT, build_model, go2_z1_official as go2_z1  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, ik_arm, initial_state, mat2quat  # noqa: E402

DT = 0.002
DECIM = 10                       # 50 Hz policy
STAIR_X0, RUN, RISE, NSTEP = 3.0, 0.30, 0.10, 6
TOP = RISE * NSTEP
PLAT_X0, PLAT_LEN = STAIR_X0 + RUN * NSTEP, 3.0
X_GOAL = 6.0                     # base x where the robot stops
WALL_Y, WALL_T, WALL_H = 0.58, 0.10, 1.0  # face 0.58 m left: wrist mid-reach with the tool horizontal
# Z1 reach with the tool pointing sideways (Monte Carlo): tip 0.45..1.0 m above the platform at y 0.3..0.6 m
# (the shoulder pitch cannot swing the upper arm below horizontal) -> crack at 0.52..0.70 m above the platform
CRACK_X, CRACK_Z0, CRACK_Z1 = X_GOAL + 0.17, TOP + 0.52, TOP + 0.70
STANDOFF = 0.012
# tool x-axis pointing +y (into the wall)
R_TOOL_WALL = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def crack_point(s):
    """s in [0,1] along the crack (bottom -> top): slight slant + zig-zag, on the wall face."""
    z = CRACK_Z0 + s * (CRACK_Z1 - CRACK_Z0)
    x = CRACK_X + 0.02 * s + 0.012 * np.sin(2 * np.pi * 3.0 * s)
    return np.array([x, WALL_Y, z])


def add_course(spec):
    wb = spec.worldbody

    def box(center, half, rgba, contact=True):
        g = wb.add_geom()
        g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_BOX, list(half), list(center), rgba
        if not contact:
            g.contype = g.conaffinity = 0

    stone = [0.62, 0.62, 0.66, 1]
    for i in range(NSTEP):
        h = RISE * (i + 1)
        box([STAIR_X0 + RUN * i + RUN / 2, 0, h / 2], [RUN / 2, 1.0, h / 2], stone)
    box([PLAT_X0 + PLAT_LEN / 2, 0, TOP / 2], [PLAT_LEN / 2, 1.0, TOP / 2], stone)
    box([PLAT_X0 + 0.4 + 2.2 / 2, WALL_Y + WALL_T / 2, TOP + WALL_H / 2], [2.2 / 2, WALL_T / 2, WALL_H / 2], [0.72, 0.66, 0.58, 1])
    pts = [crack_point(s) for s in np.linspace(0, 1, 60)]
    for a, b in zip(pts[:-1], pts[1:]):  # crack drawn as thin dark capsules just in front of the face
        g = wb.add_geom()
        g.type, g.size = mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0025, 0, 0]
        g.fromto = [a[0], a[1] - 0.002, a[2], b[0], b[1] - 0.002, b[2]]
        g.rgba, g.contype, g.conaffinity = [0.08, 0.06, 0.05, 1], 0, 0


def make_model(cfg):
    m = build_model(cfg, torch_mass=0.0, workpiece=False, scene_fn=add_course)
    for j in cfg.leg_joints:  # legs: joint dynamics the RL policies were trained with
        dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        m.dof_damping[dof], m.dof_frictionloss[dof] = 0.0, 0.01
    return m


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def fd(fun, t, h=1e-3):
    f0, fp, fm = fun(t), fun(t + h), fun(t - h)
    return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ----------------------------------------------------------------------------------------------- RL policy
class RLPolicy:
    def __init__(self, name, m, d, ix):
        self.c = POLICIES[name]
        self.m, self.d, self.ix = m, d, ix
        self.pol = torch.jit.load(os.path.join(ROOT, "policies", "go2", self.c["file"]), map_location="cpu").eval()
        self.q0 = ix.leg_q0
        self.last_a = np.zeros(12)
        self.target = self.q0.copy()
        self.primed = False

    def update(self, cmd):
        d, c, ix = self.d, self.c, self.ix
        R = d.xmat[ix.base].reshape(3, 3)
        o = np.concatenate([d.qvel[3:6] * 0.25, R.T @ np.array([0, 0, -1.0]), cmd * np.array(c["cmd_scale"]),
                            d.qpos[ix.leg_q] - self.q0, d.qvel[ix.leg_v] * 0.05, self.last_a])
        o = torch.tensor(o, dtype=torch.float32)[None]
        with torch.no_grad():
            if not self.primed:
                for _ in range(9):
                    self.pol(o)
                self.primed = True
            a = np.clip(self.pol(o)[0].numpy().astype(float), -100, 100)
        self.last_a = a
        self.target = self.q0 + c["act_scale"] * a

    def leg_torque(self):
        d, ix = self.d, self.ix
        return self.c["kp"] * (self.target - d.qpos[ix.leg_q]) - self.c["kd"] * d.qvel[ix.leg_v]


def arm_hold_torque(d, ix, q_ref):
    """Z1 held folded during locomotion: PD + gravity/Coriolis compensation."""
    ia = ix.arm_v
    return 150.0 * (q_ref - d.qpos[ix.arm_q]) - 4.0 * d.qvel[ia] + (d.qfrc_bias - d.qfrc_passive)[ia]


# ----------------------------------------------------------------------------------------------- WBC weld plan
class WeldPlan:
    """Reference generator for the stance phase, times relative to the hand-over instant th."""
    T_BLEND, T_BRACE, T_UNFOLD, T_APPROACH, T_WELD, T_RETRACT, T_STOW, T_HOLD = 0.5, 1.0, 2.5, 1.0, 8.0, 1.0, 2.5, 1.0

    def __init__(self, cfg, cm, qpos_h, th):
        self.ix = Indexer(cm, cfg)
        self.th = th
        self.base_pos0, self.base_quat0 = qpos_h[0:3].copy(), qpos_h[3:7].copy()
        self.base_brace = self.base_pos0 - [0, 0, cfg.brace_dz]
        self.stow = self.ix.arm_stow
        self.ee_quat = mat2quat(R_TOOL_WALL)
        self.p_start = crack_point(0.0) - [0, STANDOFF, 0]
        self.p_end = crack_point(1.0) - [0, STANDOFF, 0]
        self.hover_start, self.hover_end = self.p_start - [0, 0.06, 0], self.p_end - [0, 0.06, 0]
        # IK for the hover pose from the braced, *actual* stopped base pose (multi-start, max joint-limit margin)
        q_ik = qpos_h.copy(); q_ik[0:3] = self.base_brace
        ix, rng = self.ix, np.random.default_rng(0)
        # Sideways tool poses are a narrow part of the Z1 workspace: seed the IK with sampled arm configurations whose
        # tool already points (within 25 deg) along the target axis, closest to the target first.
        dm = mujoco.MjData(cm); dm.qpos[:] = q_ik
        cand = []
        for _ in range(30000):
            qa = rng.uniform(ix.arm_lo, ix.arm_hi)
            dm.qpos[ix.arm_q] = qa
            mujoco.mj_kinematics(cm, dm)
            if dm.site_xmat[ix.tip].reshape(3, 3)[:, 0] @ R_TOOL_WALL[:, 0] > np.cos(np.radians(25)):
                cand.append((np.linalg.norm(dm.site_xpos[ix.tip] - self.hover_start), qa))
        cand.sort(key=lambda c: c[0])
        seeds = [c[1] for c in cand[:30]] + [np.array(s, float) for s in cfg.ik_seeds]
        span, best, best_score = ix.arm_hi - ix.arm_lo, None, -np.inf
        self.ik_debug = dict(n_candidates=len(cand), nearest_sample_mm=round(1000 * cand[0][0], 1) if cand else None,
                             base=np.round(qpos_h[0:3], 3).tolist(), hover_target=np.round(self.hover_start, 3).tolist())
        res_all = []
        for s in seeds:
            q, res = ik_arm(cm, q_ik, ix, self.hover_start, R_TOOL_WALL, seed=s, iters=400, axis_only=True)
            res_all.append(res)
            score = np.min(np.minimum(q - ix.arm_lo, ix.arm_hi - q) / span)
            if res < 1e-3 and score > best_score:
                best, best_score = q, score
        if best is None:
            self.ik_debug["best_ik_residual_mm"] = round(1000 * min(res_all), 1)
            raise RuntimeError(f"crack not reachable from where the robot stopped: {self.ik_debug}")
        self.q_hover = best
        # tool orientation the WBC holds = the one the (roll-free) IK found, so the 6-D tool task is consistent
        dm.qpos[:] = q_ik; dm.qpos[ix.arm_q] = best; mujoco.mj_kinematics(cm, dm)
        self.ee_quat = mat2quat(dm.site_xmat[ix.tip])
        self.ik_res_mm = 1000 * max(ik_arm(cm, q_ik, ix, p, R_TOOL_WALL, seed=best, axis_only=True)[1] for p in (self.p_start, self.p_end))
        self.q_restow_start = None
        k = np.cumsum([0, self.T_BLEND, self.T_BRACE, self.T_UNFOLD, self.T_APPROACH, self.T_WELD, self.T_RETRACT, self.T_STOW, self.T_HOLD])
        self.t_marks = dict(zip(["blend", "brace", "unfold", "approach", "weld", "retract", "stow", "hold", "end"], k))

    def phase(self, t):
        tau, name = t - self.th, "blend"
        for n, t0 in self.t_marks.items():
            if tau >= t0:
                name = n
        return name

    def _ee(self, tau):
        T = self.t_marks
        if tau < T["weld"]:
            return self.hover_start + min_jerk((tau - T["approach"]) / self.T_APPROACH) * (self.p_start - self.hover_start)
        if tau < T["retract"]:
            Tw, Tr, u = self.T_WELD, 0.5, tau - T["weld"]  # constant travel speed, smooth ramps
            vmax = 1.0 / (Tw - Tr)
            I = lambda s: 2.5 * s**4 - 3 * s**5 + s**6
            s = vmax * Tr * I(u / Tr) if u < Tr else (1.0 - vmax * Tr * I((Tw - u) / Tr) if u > Tw - Tr else vmax * (u - Tr / 2))
            return crack_point(s) - [0, STANDOFF, 0]
        return self.p_end + min_jerk((tau - T["retract"]) / self.T_RETRACT) * (self.hover_end - self.p_end)

    def ref(self, t, arm_q_now):
        tau, T = t - self.th, self.t_marks
        base = self.base_pos0 + min_jerk((tau - T["brace"]) / self.T_BRACE) * (self.base_brace - self.base_pos0)
        r = dict(base_pos=base, base_quat=self.base_quat0, w_ee=0.0, w_post=10.0)
        if T["approach"] <= tau < T["stow"]:
            p, v, a = fd(self._ee, tau)
            n = len(self.stow)
            r.update(ee_pos=p, ee_vel=v, ee_acc=a, ee_quat=self.ee_quat, w_ee=30.0, w_post=1e-4,
                     arm_q=self.q_hover, arm_qd=np.zeros(n), arm_qdd=np.zeros(n))
            return r
        if tau >= T["stow"]:
            if self.q_restow_start is None:
                self.q_restow_start = arm_q_now.copy()
            f = lambda x: self.q_restow_start + min_jerk((x - T["stow"]) / self.T_STOW) * (self.stow - self.q_restow_start)
        else:
            f = lambda x: self.stow + min_jerk((x - T["unfold"]) / self.T_UNFOLD) * (self.q_hover - self.stow)
        q, qd, qdd = fd(f, tau)
        r.update(arm_q=q, arm_qd=qd, arm_qdd=qdd)
        return r


# ----------------------------------------------------------------------------------------------- mission
def run_mission(policy, render=True, t_max=60.0):
    cfg = go2_z1()
    m, cm = make_model(cfg), make_model(cfg)
    d = mujoco.MjData(m)
    ix = Indexer(m, cfg)
    initial_state(m, d, ix)
    rl = RLPolicy(policy, m, d, ix)
    wbc, plan = None, None
    mode, t_stop, th = "walk", None, None
    tau_rl_frozen = None

    frames, bead, log = [], [], dict(t=[], pos=[], quat=[], mode=[], tip=[], tip_ref=[])
    ren = cam = None
    if render:
        ren = mujoco.Renderer(m, 480, 854)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, ix.base
        cam.distance, cam.azimuth, cam.elevation = 2.4, 60, -20

    events = {}
    for k in range(int(t_max / DT)):
        t = d.time
        x, y = d.qpos[0], d.qpos[1]
        yaw = yaw_of(d.qpos[3:7])
        R = d.xmat[ix.base].reshape(3, 3)
        if R[2, 2] < 0.5:
            events["fell_at"] = round(t, 2)
            break

        tau = np.zeros(m.nv)
        if mode in ("walk", "stop"):
            if k % DECIM == 0:
                if mode == "walk":
                    vx = 0.5 if x < X_GOAL - 0.4 else max(0.15, 0.5 * (X_GOAL - x) / 0.4)
                    wz = float(np.clip(-2.0 * yaw - 1.5 * y, -0.5, 0.5))  # heading + lane keeping
                    cmd = np.array([vx, 0.0, wz])
                    if x >= X_GOAL:
                        mode, t_stop = "stop", t
                        events["reached_goal_s"] = round(t, 2)
                else:
                    cmd = np.zeros(3)
                rl.update(cmd)
                if "on_platform_s" not in events and d.qpos[2] > TOP + 0.2 and x > PLAT_X0:
                    events["on_platform_s"] = round(t, 2)
            tau[ix.leg_v] = rl.leg_torque()
            tau[ix.arm_v] = arm_hold_torque(d, ix, ix.arm_stow)
            if mode == "stop" and t - t_stop > 1.5 and np.linalg.norm(d.qvel[0:3]) < 0.05 and np.linalg.norm(d.qvel[3:6]) < 0.15:
                mode, th = "wbc", t  # ---- hand-over
                plan = WeldPlan(cfg, cm, d.qpos.copy(), th)
                wbc = StanceWBC(cm, cfg, **cfg.wbc_gains)
                events.update(handover_s=round(th, 2), stop_pose=[round(x, 3), round(y, 3), round(float(np.degrees(yaw)), 2)],
                              crack_ik_residual_mm=round(plan.ik_res_mm, 4))
        if mode == "wbc":
            ref = plan.ref(t, d.qpos[ix.arm_q].copy())
            tau_w, f = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
            s = np.clip((t - th) / WeldPlan.T_BLEND, 0, 1)
            if s < 1:  # RL leg targets frozen at hand-over, arm hold, blended into the WBC torques
                tau_r = np.zeros(m.nv)
                tau_r[ix.leg_v] = rl.leg_torque()
                tau_r[ix.arm_v] = arm_hold_torque(d, ix, ix.arm_stow)
                tau = (1 - s) * tau_r + s * tau_w
            else:
                tau = tau_w
            if plan.phase(t) == "end":
                break
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(m, d)

        ph = plan.phase(d.time) if mode == "wbc" else mode
        log["t"].append(d.time); log["pos"].append(d.qpos[0:3].copy()); log["quat"].append(d.qpos[3:7].copy()); log["mode"].append(ph)
        log["tip"].append(d.site_xpos[ix.tip].copy())
        log["tip_ref"].append(ref["ee_pos"] if mode == "wbc" and "ee_pos" in ref else np.full(3, np.nan))
        if ren is not None and k % DECIM == DECIM - 1:
            if mode == "wbc":  # move in closer for the weld
                a = min_jerk((d.time - th) / 2.0)
                cam.distance, cam.azimuth, cam.elevation = 2.4 - 0.9 * a, 60 - 15 * a, -20 + 5 * a
            ren.update_scene(d, camera=cam)
            if ph == "weld":
                bead.append(d.site_xpos[ix.tip].copy() + [0, STANDOFF - 0.004, 0])
            scn = ren.scene
            for p in bead:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p,
                                    np.eye(3).flatten(), np.array([1.0, 0.45, 0.05, 1], np.float32))
                scn.ngeom += 1
            ctrl = POLICIES[policy]["label"] if mode in ("walk", "stop") else ("RL->WBC blend" if ph == "blend" else "stance WBC")
            frames.append(overlay(ren.render(), f"[{ph}]  {ctrl}  t={d.time:5.1f}s  x={x:5.2f} m", log, plan, th))
    L = {k: np.array(v) for k, v in log.items() if k != "mode"}
    modes = np.array(log["mode"])
    res = dict(policy=policy, **events, completed=bool(plan is not None and plan.phase(d.time) == "end" and "fell_at" not in events))
    if plan is not None:
        w = modes == "weld"
        i0 = int(np.argmax(modes == "unfold"))
        e = 1000 * np.linalg.norm(L["tip"][w] - L["tip_ref"][w], axis=1)
        res.update(weld_tool_err_rms_mm=float(np.sqrt(np.mean(e**2))), weld_tool_err_max_mm=float(e.max()),
                   base_drift_during_arm_mm=float(1000 * np.max(np.linalg.norm(L["pos"][i0:] - L["pos"][i0], axis=1))),
                   base_tilt_during_arm_deg=float(max(rot_deg(L["quat"][i0], q) for q in L["quat"][i0:])),
                   handover_base_jump_mm=float(1000 * np.max(np.linalg.norm(L["pos"][modes == "blend"] - L["pos"][modes == "blend"][0], axis=1))))
    return res, frames


def rot_deg(q0, q):
    e = np.zeros(3); mujoco.mju_subQuat(e, q, q0)
    return float(np.degrees(np.linalg.norm(e)))


def overlay(img, title, log, plan, th):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    line2 = ""
    if plan is not None and not np.isnan(log["tip_ref"][-1][0]):
        line2 = f"tool err {1000 * np.linalg.norm(log['tip'][-1] - log['tip_ref'][-1]):5.2f} mm"
    dr.rectangle([0, 0, img.shape[1], 36], fill=(0, 0, 0))
    dr.multiline_text((6, 3), title + ("\n" + line2 if line2 else ""), fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="moe_cts,symmetry")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "course")
    os.makedirs(out, exist_ok=True)
    import imageio
    rows = []
    for p in args.policy.split(","):
        res, frames = run_mission(p, render=not args.no_video)
        print(json.dumps(res)); rows.append(res)
        if frames:
            imageio.mimsave(os.path.join(out, f"mission__{p}.mp4"), frames, fps=50, quality=7)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
