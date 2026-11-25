"""Stance WBC demo: the quadruped stands still, the arm welds a seam, the base must stay rigid.

Robot (sim/wbc/robots.py):  go2_z1 (Unitree Go2 + Z1; no torch by default - massless tool point past the flange)
Scenarios:
    weld   : brace (lower base 4 cm) -> unstow -> descend -> weld 30 cm seam with 4 mm weave -> retract
    stress : brace -> unstow -> large fast arm sweeps (worst-case CoM excursions)

    python sim/stance_demo.py                                         # both scenarios -> results/stance_wbc/go2_z1/
    python sim/stance_demo.py --scenario weld
    python sim/stance_demo.py --torch-mass 1.2 --ctrl-torch-mass 0.6     # with a torch; controller believes the wrong mass
    mjpython sim/stance_demo.py --scenario weld --viewer
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wbc.robots import ROBOTS, ROOT, build_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, ik_arm, initial_state, mat2quat  # noqa: E402

DT = 0.002
R_TORCH_DOWN = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # torch (tool x-axis) pointing at the floor


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def fd(fun, t, h=1e-3):
    """value, 1st and 2nd time derivative by central differences (references are smooth)."""
    f0, fp, fm = fun(t), fun(t + h), fun(t - h)
    return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h**2


# ----------------------------------------------------------------------------------------------- task plan
class Plan:
    STANDOFF = 0.012

    def __init__(self, scenario, cfg, ctrl_model, qpos0):
        self.scn, self.cfg = scenario, cfg
        ix = Indexer(ctrl_model, cfg)
        self.stow = ix.arm_stow
        self.base_pos0, self.base_quat0 = qpos0[0:3].copy(), qpos0[3:7].copy()
        self.base_brace = self.base_pos0 - np.array([0, 0, cfg.brace_dz])
        self.z_top = 2 * cfg.workpiece_half[2]
        z_weld = self.z_top + self.STANDOFF
        self.p_start = np.array([cfg.workpiece_x, -cfg.seam_half, z_weld])
        self.p_end = np.array([cfg.workpiece_x, cfg.seam_half, z_weld])
        self.hover_start = self.p_start + [0, 0, 0.08]
        self.hover_end = self.p_end + [0, 0, 0.08]
        self.ee_quat = mat2quat(R_TORCH_DOWN)

        q_ik = qpos0.copy(); q_ik[0:3] = self.base_brace
        # multi-start IK; among (near-)exact solutions keep the one furthest from joint limits (the WBC does not
        # model joint-limit forces, so a hover pose near a limit would disturb the base)
        rng = np.random.default_rng(0)
        seeds = [np.array(s, float) for s in cfg.ik_seeds]
        seeds += [np.concatenate([rng.uniform(0.9 * ix.arm_lo[:6], 0.9 * ix.arm_hi[:6]), self.stow[6:]]) for _ in range(20)]
        span = ix.arm_hi[:6] - ix.arm_lo[:6]
        best, best_score = None, -np.inf
        for seed in seeds:
            q, res = ik_arm(ctrl_model, q_ik, ix, self.hover_start, R_TORCH_DOWN, seed=seed, iters=200)
            margin = np.min(np.minimum(q[:6] - ix.arm_lo[:6], ix.arm_hi[:6] - q[:6]) / span)
            score = margin - 10.0 * res  # 1 mm residual ~ 1% of range margin
            if res < 1e-3 and score > best_score:
                best, best_score = (q, res), score
        if best is None:
            raise RuntimeError("seam hover pose unreachable from the braced stance - move the workpiece")
        self.q_hover, res = best
        if ix.n_arm > 6:
            self.q_hover[6:] = self.stow[6:]  # gripper stays closed
        # residual of the IK at the seam ends: tells whether the whole seam is reachable from the braced stance
        self.ik_residual_mm = 1000 * max(res, *(ik_arm(ctrl_model, q_ik, ix, p, R_TORCH_DOWN, seed=self.q_hover)[1]
                                                for p in (self.p_start, self.p_end)))
        self.sweep_amp = np.array(cfg.sweep_amp, float)
        self.q_lo, self.q_hi = ix.arm_lo, ix.arm_hi

        if scenario == "weld":
            self.T = 16.0
            self.phases = [(0.0, "settle"), (1.0, "brace"), (2.0, "unstow"), (4.5, "descend"),
                           (5.5, "weld"), (13.5, "retract"), (14.5, "hold")]
            self.window = (5.5, 13.5)
        else:
            self.T = 15.0
            self.phases = [(0.0, "settle"), (1.0, "brace"), (2.0, "unstow"), (4.5, "sweep"), (13.5, "hold")]
            self.window = (4.5, 13.5)
        self.arm_start = 2.0

    def phase(self, t):
        name = self.phases[0][1]
        for t0, n in self.phases:
            if t >= t0:
                name = n
        return name

    def _base(self, t):
        return self.base_pos0 + min_jerk(t - 1.0) * (self.base_brace - self.base_pos0)

    def _arm_joint(self, t):
        q = self.stow + min_jerk((t - 2.0) / 2.5) * (self.q_hover - self.stow)
        if self.scn == "stress" and t > 4.5:
            tau = t - 4.5
            amp = min_jerk(tau / 1.0) * (1 - min_jerk((t - 12.5) / 1.0))
            freqs = np.array([0.4, 0.6, 0.5, 0.7, 0.55, 0.65, 0.5])[:len(q)]
            # keep the sweep inside the joint ranges (0.1 rad margin)
            a = np.minimum(self.sweep_amp, np.maximum(0.0, np.minimum(self.q_hi - 0.1 - self.q_hover, self.q_hover - self.q_lo - 0.1)))
            q = q + amp * a * np.sin(2 * np.pi * freqs * tau)
        return q

    def _ee(self, t):
        if t < 5.5:
            return self.hover_start + min_jerk(t - 4.5) * (self.p_start - self.hover_start)
        if t < 13.5:
            # constant travel speed with C2-smooth (min-jerk velocity) ramps at both ends
            T, Tr, tau = 8.0, 0.5, t - 5.5
            vmax = 1.0 / (T - Tr)
            I = lambda s: 2.5 * s**4 - 3 * s**5 + s**6  # integral of min_jerk
            if tau < Tr:
                s = vmax * Tr * I(tau / Tr)
            elif tau > T - Tr:
                s = 1.0 - vmax * Tr * I((T - tau) / Tr)
            else:
                s = vmax * (tau - Tr / 2)
            p = self.p_start + s * (self.p_end - self.p_start)
            # weave: +-4 mm across the seam at 1.5 Hz, faded in/out over 0.5 s
            fade = min_jerk((t - 5.5) / 0.5) * (1 - min_jerk((t - 13.0) / 0.5))
            return p + np.array([0.004 * fade * np.sin(2 * np.pi * 1.5 * (t - 5.5)), 0, 0])
        return self.p_end + min_jerk(t - 13.5) * (self.hover_end - self.p_end)

    def ref(self, t):
        q, qd, qdd = fd(self._arm_joint, t)
        r = dict(base_pos=self._base(t), base_quat=self.base_quat0, arm_q=q, arm_qd=qd, arm_qdd=qdd, w_ee=0.0, w_post=10.0)
        if self.scn == "weld" and t >= 4.5:
            p, v, a = fd(self._ee, t)
            n = len(q)
            r.update(ee_pos=p, ee_vel=v, ee_acc=a, ee_quat=self.ee_quat, w_ee=30.0, w_post=1e-4,
                     arm_q=self.q_hover, arm_qd=np.zeros(n), arm_qdd=np.zeros(n))
        return r


# ----------------------------------------------------------------------------------------------- episode
def run(robot, scenario, torch_mass, ctrl_torch_mass, render=True, viewer=False, cfg=None, wbc_kw=None, wbc_hook=None,
        label=None):
    """cfg / wbc_kw / wbc_hook / label: overrides for tuning studies (sim/wbc_tuning_study.py); None = the tuned WBC."""
    cfg = cfg or ROBOTS[robot]()
    plant = build_model(cfg, torch_mass=torch_mass)
    ctrl_model = build_model(cfg, torch_mass=ctrl_torch_mass)
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    initial_state(plant, d, ix)
    plan = Plan(scenario, cfg, ctrl_model, d.qpos.copy())
    wbc = StanceWBC(ctrl_model, cfg, **dict(cfg.wbc_gains, **(wbc_kw or {})))
    if wbc_hook is not None:
        wbc_hook(wbc)
    n = int(plan.T / DT)

    renderer = cam = None
    frames, bead = [], []
    if render:
        renderer = mujoco.Renderer(plant, 480, 854)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = cfg.cam["lookat"]
        cam.distance, cam.azimuth, cam.elevation = cfg.cam["distance"], cfg.cam["azimuth"], cfg.cam["elevation"]
    v = None
    if viewer:
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch_passive(plant, d)

    L = {k: [] for k in ["t", "base_pos", "base_quat", "v", "w", "tip", "tip_ref", "feet", "f", "tau"]}
    solve_ms = []
    for k in range(n):
        t = d.time
        ref = plan.ref(t)
        t0 = time.perf_counter()
        tau, f = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - t0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        L["t"].append(d.time); L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
        L["v"].append(d.qvel[0:3].copy()); L["w"].append(d.qvel[3:6].copy()); L["tip"].append(d.site_xpos[ix.tip].copy())
        L["tip_ref"].append(ref["ee_pos"] if "ee_pos" in ref else np.full(3, np.nan))
        L["feet"].append(d.geom_xpos[ix.feet].copy()); L["f"].append(f.copy()); L["tau"].append(d.actuator_force.copy())
        if v is not None:
            if not v.is_running():
                break
            v.sync()
            time.sleep(max(0.0, DT - 1e-4))
        if renderer is not None and k % 10 == 9:
            renderer.update_scene(d, camera=cam)
            tip = d.site_xpos[ix.tip]
            if plan.phase(t) == "weld" and tip[2] < plan.z_top + 0.03:
                bead.append(tip.copy() - [0, 0, 0.008])
            scn = renderer.scene
            for p in bead[-1500:]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.0045, 0, 0], p,
                                    np.eye(3).flatten(), np.array([1.0, 0.45, 0.05, 1], np.float32))
                scn.ngeom += 1
            frames.append(overlay(renderer.render(), label or robot, scenario, t, plan.phase(t), L, plan))
    if v is not None:
        v.close()
    L = {k: np.array(x) for k, x in L.items()}
    res = metrics(L, plan, robot, scenario, wbc, solve_ms, torch_mass, ctrl_torch_mass, plant)
    return res, L, frames, plan


def rot_dev_deg(q_ref, Q):
    out = []
    for q in Q:
        e = np.zeros(3); mujoco.mju_subQuat(e, q, q_ref); out.append(np.degrees(np.linalg.norm(e)))
    return np.array(out)


def overlay(img, robot, scn, t, phase, L, plan):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    i0 = int(plan.arm_start / DT)
    dev = 1000 * np.linalg.norm(L["base_pos"][-1] - L["base_pos"][i0]) if len(L["base_pos"]) > i0 else 0.0
    txt = f"WBC  {robot}  |  {scn}  t={t:4.1f}s  [{phase}]\nbase drift {dev:6.3f} mm"
    if not np.isnan(L["tip_ref"][-1][0]):
        txt += f"   torch err {1000 * np.linalg.norm(L['tip'][-1] - L['tip_ref'][-1]):6.3f} mm"
    dr.rectangle([0, 0, img.shape[1], 38], fill=(0, 0, 0))
    dr.multiline_text((6, 3), txt, fill=(255, 255, 255))
    return np.asarray(im)


def metrics(L, plan, robot, scn, wbc, solve_ms, torch_mass, ctrl_torch_mass, plant):
    t = L["t"]
    i0 = int(plan.arm_start / DT)
    win = (t >= plan.window[0]) & (t <= plan.window[1])
    after = t >= plan.arm_start
    pos_dev = 1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1)
    rot_dev = rot_dev_deg(L["base_quat"][i0], L["base_quat"])
    slip = 1000 * np.max(np.linalg.norm((L["feet"] - L["feet"][i0])[:, :, :2], axis=2), axis=1)
    tip_err = 1000 * np.linalg.norm(L["tip"] - L["tip_ref"], axis=1)
    r = dict(robot=robot, scenario=scn, torch_mass=torch_mass, ctrl_torch_mass=ctrl_torch_mass,
             fell=bool(np.any(L["base_pos"][:, 2] < 0.2)),
             base_pos_dev_max_mm=float(pos_dev[after].max()), base_rot_dev_max_deg=float(rot_dev[after].max()),
             base_vel_rms_mm_s=float(1000 * np.sqrt(np.mean(np.sum(L["v"][win] ** 2, 1)))),
             base_angvel_rms_deg_s=float(np.degrees(np.sqrt(np.mean(np.sum(L["w"][win] ** 2, 1))))),
             foot_slip_max_mm=float(slip[after].max()),
             peak_torque_frac=float(np.max(np.abs(L["tau"][after]) / plant.actuator_forcerange[:, 1])),
             seam_ik_residual_mm=float(plan.ik_residual_mm),
             qp_ms_mean=float(np.mean(solve_ms)), qp_ms_max=float(np.max(solve_ms)), qp_failures=wbc.fail)
    if scn == "weld":
        r["torch_err_rms_mm"] = float(np.sqrt(np.nanmean(tip_err[win] ** 2)))
        r["torch_err_max_mm"] = float(np.nanmax(tip_err[win]))
    return r


def plot(L, plan, robot, scn, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = 4 if scn == "weld" else 3
    fig, ax = plt.subplots(rows, 1, figsize=(10, 2.5 * rows), sharex=True)
    i0 = int(plan.arm_start / DT)
    t = L["t"]
    ax[0].plot(t, 1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1), color="#1f6feb")
    ax[0].set_ylabel("base drift [mm]")
    ax[1].plot(t, rot_dev_deg(L["base_quat"][i0], L["base_quat"]), color="#1f6feb")
    ax[1].set_ylabel("base rot dev [deg]")
    if scn == "weld":
        ax[2].plot(t, 1000 * np.linalg.norm(L["tip"] - L["tip_ref"], axis=1), color="#cf222e")
        ax[2].set_ylabel("torch tip err [mm]")
    for k, nm in enumerate(["LF", "RF", "LH", "RH"]):
        ax[-1].plot(t, L["f"][:, 3 * k + 2], label=nm)
    ax[-1].set_ylabel("foot normal force [N]"); ax[-1].legend(ncol=4, fontsize=8)
    for a in ax:
        for t0, nm in plan.phases:
            a.axvline(t0, color="0.8", lw=0.8)
        a.grid(alpha=0.3)
    for t0, nm in plan.phases:
        ax[0].text(t0 + 0.05, 0.9, nm, fontsize=7, color="0.4", transform=ax[0].get_xaxis_transform())
    ax[-1].set_xlabel("time [s]")
    fig.suptitle(f"Stance WBC — {robot} — {scn}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="go2_z1_official")
    ap.add_argument("--scenario", choices=["weld", "stress", "all"], default="all")
    ap.add_argument("--torch-mass", type=float, default=0.0, help="true torch mass in the plant [kg] (0: no torch)")
    ap.add_argument("--ctrl-torch-mass", type=float, default=None, help="torch mass the controller believes [kg]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ctm = args.torch_mass if args.ctrl_torch_mass is None else args.ctrl_torch_mass
    out = os.path.join(ROOT, "results", "stance_wbc", args.robot)
    os.makedirs(out, exist_ok=True)
    scns = ["weld", "stress"] if args.scenario == "all" else [args.scenario]
    rows = []
    for scn in scns:
        res, L, frames, plan = run(args.robot, scn, args.torch_mass, ctm, render=not (args.no_video or args.viewer), viewer=args.viewer)
        print(json.dumps(res))
        rows.append(res)
        plot(L, plan, args.robot, scn, os.path.join(out, f"{scn}{args.tag}.png"))
        if frames:
            import imageio
            imageio.mimsave(os.path.join(out, f"{scn}{args.tag}.mp4"), frames, fps=50, quality=7)
    with open(os.path.join(out, f"summary{args.tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["scenario", "fell", "base_pos_dev_max_mm", "base_rot_dev_max_deg", "base_vel_rms_mm_s", "base_angvel_rms_deg_s",
            "torch_err_rms_mm", "torch_err_max_mm", "foot_slip_max_mm", "peak_torque_frac", "seam_ik_residual_mm", "qp_ms_mean", "qp_failures"]
    fmt = lambda x: "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    with open(os.path.join(out, f"summary{args.tag}.md"), "w") as f:
        f.write(f"robot={args.robot}  torch mass plant={args.torch_mass} kg, controller model={ctm} kg\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
