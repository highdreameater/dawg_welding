"""Freeform-surface demo: the robot stands at the "panel" station, scans an unknown random surface with the wrist
camera, reconstructs it (thin-plate spline, no parametric shape) and traces an S-path across it with the torch tip
pressing a constant normal force (see sim/surface_task.py).

    python sim/surface_demo.py                                   # Go2 + Z1, 5 N, random panel seed 0
    python sim/surface_demo.py --panel-seed 3 --panel-amp 0.04   # another / rougher surface
    python sim/surface_demo.py --perception flat                 # no scan: assume a flat panel at the map height
    python sim/surface_demo.py --robot anymal_c_z1 --force 8

All accuracy metrics are measured against the true (analytic) surface. -> results/surface/
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from force_demo import ForceSensor, Recorder, draw_cloud, rot_dev_deg  # noqa: E402
from perception.depth_camera import CameraMount, DepthCamera  # noqa: E402
from perception.surface_fit import FlatSurface, fit_surface  # noqa: E402
from surface_task import SurfaceTraceTask  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import WORLDS, panel_dock_pose  # noqa: E402

DT = 0.002
WRIST_CAM = CameraMount(body="z1_link06", pos=(0.06, 0.0, 0.05), pitch_deg=0.0)  # beside the torch, along the tool


def run(args, cfg, world, render=True):
    pn = world.panels[0]
    prior = (pn.cx, pn.cy, pn.hx, pn.hy, pn.top_z)   # the map: stand footprint + nominal height, not the shape
    plant = build_model(cfg, world=world, contact_tip=True)
    ctrl = controller_model(cfg)
    plant.opt.timestep = ctrl.opt.timestep = DT
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    dock = panel_dock_pose(pn, cfg.pipe_work_dist)
    initial_state(plant, d, ix, xy=dock[:2], yaw=dock[2])
    task = SurfaceTraceTask(cfg, ctrl, d.qpos.copy(), prior, f_des=args.force, passes=args.passes, speed=args.speed)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    cam = DepthCamera(plant, WRIST_CAM)
    rec = Recorder(plant, dict(lookat=[pn.cx - 0.12, pn.cy, pn.top_z + 0.1], distance=1.25, azimuth=215,
                               elevation=-32)) if render else None
    clouds, info = [], dict(perception=args.perception)
    path_marks = None
    L = {k: [] for k in ["t", "f_raw", "f_des", "phase", "tip", "axis", "n_cmd", "u", "base_pos", "base_quat", "arm_q",
                         "feet", "tau"]}
    solve_ms = []
    k = 0
    while d.time < task.t0 + task.T:
        t = d.time
        ph = task.phase(t)
        if ph == "scan" and t - task.t0 > task.start["scan"] + 0.4 and k % 10 == 0:
            clouds.append(cam.capture(plant, d)[0])
        if task.model is None and ph not in ("settle", "brace", "to_scan", "scan"):
            c0 = time.perf_counter()
            if args.perception == "wrist":
                model = fit_surface(np.concatenate(clouds), prior[:4])
                info.update(frames=len(clouds), raw_points=int(sum(len(c) for c in clouds)),
                            model={k_: v for k_, v in model.info.items()})
            else:
                model = FlatSurface(pn.top_z)
            task.set_model(model)
            info.update(model_ms=1000 * (time.perf_counter() - c0), path=task.path_choice)
            path_marks = np.array([task.path_point(u)[0] for u in np.linspace(0, 1, 60)])
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas, sensor.vec.copy())
        c0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - c0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        tr = t - task.t0
        L["t"].append(tr); L["f_raw"].append(sensor.raw); L["phase"].append(task.phase(t))
        L["f_des"].append(ref["force"]["f_des"] if "force" in ref else np.nan)
        L["tip"].append(d.site_xpos[ix.tip].copy()); L["axis"].append(d.site_xmat[ix.tip].reshape(3, 3)[:, 0].copy())
        L["n_cmd"].append(ref["force"]["n"] if "force" in ref else np.full(3, np.nan)); L["u"].append(task.u(tr))
        L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
        L["arm_q"].append(d.qpos[ix.arm_q].copy())
        L["feet"].append(d.geom_xpos[ix.feet].copy()); L["tau"].append(d.actuator_force.copy())
        if rec is not None and k % 10 == 9:
            tip = d.site_xpos[ix.tip]
            dev = 1000 * (tip[2] - pn.height(tip[0], tip[1]))
            txt = (f"freeform surface  {cfg.name}  [{args.perception}]  t={t:5.2f}s  [{L['phase'][-1]}]\n"
                   f"F = {sensor.raw:5.2f} N (target {args.force:.1f})   tool vs true normal "
                   f"{np.degrees(np.arccos(np.clip(-L['axis'][-1] @ pn.normal(tip[0], tip[1]).reshape(3), -1, 1))):4.1f} deg")
            cloud = None
            if clouds and L["phase"][-1] in ("scan", "to_hover"):
                cloud = (clouds[-1][::3], [0.1, 1.0, 0.3, 1])
            elif path_marks is not None and L["phase"][-1] in ("hover", "approach", "sweep", "dwell"):
                cloud = (path_marks + [0, 0, 0.002], [1.0, 0.85, 0.1, 1])
            rec.frame(d, txt, tip.copy(), pn.normal(tip[0], tip[1]).reshape(3), sensor.raw, cloud)
        k += 1
    L = {k_: (np.array(x) if k_ != "phase" else np.array(x, dtype=object)) for k_, x in L.items()}
    return L, task, wbc, plant, solve_ms, (rec.frames if rec else []), info


def metrics(L, task, wbc, plant, solve_ms, pn, args, info):
    t, ph = L["t"], L["phase"]
    sweep = ph == "sweep"
    contact = np.isin(ph, ["sweep", "dwell"])
    i0 = int(task.start["to_scan"] / DT)
    after = t >= task.start["to_scan"]
    f, fe = L["f_raw"], L["f_raw"] - args.force
    tip = L["tip"]
    n_true = pn.normal(tip[:, 0], tip[:, 1])
    axis_e = np.degrees(np.arccos(np.clip(np.sum(L["axis"] * -n_true, 1), -1, 1)))
    # xy deviation of the tip from the planned path
    dense = task.path_xy(np.linspace(0, 1, 2000))
    dxy = np.array([np.min(np.linalg.norm(dense - p[:2], axis=1)) for p in tip[sweep][::5]])
    # the surface model along the path vs the truth
    pts = np.array([task.path_point(u)[0] for u in np.linspace(0, 1, 200)])
    nm = np.array([task.path_point(u)[1] for u in np.linspace(0, 1, 200)])
    h_err = 1000 * (pts[:, 2] - pn.height(pts[:, 0], pts[:, 1]))
    n_err = np.degrees(np.arccos(np.clip(np.sum(nm * pn.normal(pts[:, 0], pts[:, 1]), 1), -1, 1)))
    n_true_path = pn.normal(pts[:, 0], pts[:, 1])
    pos_dev = 1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1)
    slip = 1000 * np.max(np.linalg.norm((L["feet"] - L["feet"][i0])[:, :, :2], axis=2), axis=1)
    return dict(
        **info, robot=args.robot, panel_seed=args.panel_seed, panel_amp=args.panel_amp, f_des=args.force,
        passes=args.passes, speed_mm_s=1000 * args.speed,
        surface_slope_along_path_deg=[round(float(x), 1) for x in np.percentile(np.degrees(np.arccos(n_true_path[:, 2])), [50, 100])],
        model_height_err_along_path_mm=dict(rms=float(np.sqrt(np.mean(h_err ** 2))), max=float(np.abs(h_err).max())),
        model_normal_err_along_path_deg=dict(rms=float(np.sqrt(np.mean(n_err ** 2))), max=float(n_err.max())),
        sweep_f_mean=float(f[sweep].mean()), sweep_f_rms_err=float(np.sqrt(np.mean(fe[sweep] ** 2))),
        sweep_f_min=float(f[sweep].min()), sweep_f_max=float(f[sweep].max()),
        contact_lost_frac=float(np.mean(f[contact] < 0.05)),
        axis_err_vs_true_normal_deg=dict(rms=float(np.sqrt(np.mean(axis_e[sweep] ** 2))), max=float(axis_e[sweep].max())),
        path_xy_dev_mm=dict(rms=float(1000 * np.sqrt(np.mean(dxy ** 2))), max=float(1000 * dxy.max())),
        base_pos_dev_max_mm=float(pos_dev[after].max()),
        base_rot_dev_max_deg=float(rot_dev_deg(L["base_quat"][i0], L["base_quat"])[after].max()),
        foot_slip_max_mm=float(slip[after].max()),
        peak_torque_frac=float(np.max(np.abs(L["tau"][after]) / plant.actuator_forcerange[:, 1])),
        qp_ms_mean=float(np.mean(solve_ms)), qp_failures=wbc.fail, fell=bool(np.any(L["base_pos"][:, 2] < 0.15)))


def plot(L, task, pn, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3)
    X, Y = np.meshgrid(np.linspace(pn.cx - pn.hx, pn.cx + pn.hx, 80), np.linspace(pn.cy - pn.hy, pn.cy + pn.hy, 140))
    ext = [pn.cy - pn.hy, pn.cy + pn.hy, pn.cx - pn.hx, pn.cx + pn.hx]  # plotted with y horizontal
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(1000 * (pn.height(X, Y) - pn.top_z).T, origin="lower", extent=ext, cmap="terrain", aspect="auto")
    fig.colorbar(im, ax=ax, label="mm")
    pxy = task.path_xy(np.linspace(0, 1, 300))
    ax.plot(pxy[:, 1], pxy[:, 0], "k-", lw=1.2, label="planned path")
    sw = L["phase"] == "sweep"
    ax.plot(L["tip"][sw, 1], L["tip"][sw, 0], "r.", ms=0.6, label="tip (sweep)")
    ax.set_title("true surface height"); ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, 1])
    err = 1000 * (task.model.height(X, Y) - pn.height(X, Y))
    err = np.where(task.model.covered(X, Y), err, np.nan)
    im = ax.imshow(err.T, origin="lower", extent=ext, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    fig.colorbar(im, ax=ax, label="mm")
    ax.plot(pxy[:, 1], pxy[:, 0], "k-", lw=0.8)
    ax.set_title(f"model - true height ({args.perception})"); ax.set_xlabel("y [m]")
    ax = fig.add_subplot(gs[0, 2])
    nerr = np.degrees(np.arccos(np.clip(np.sum(task.model.normal(X, Y) * pn.normal(X, Y), -1), -1, 1)))
    im = ax.imshow(np.where(task.model.covered(X, Y), nerr, np.nan).T, origin="lower", extent=ext, cmap="magma_r",
                   vmin=0, vmax=10, aspect="auto")
    fig.colorbar(im, ax=ax, label="deg")
    ax.plot(pxy[:, 1], pxy[:, 0], "k-", lw=0.8)
    ax.set_title("model normal error"); ax.set_xlabel("y [m]")
    t = L["t"]
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, L["f_raw"], color="#cf222e", lw=0.8, label="measured")
    ax.plot(t, L["f_des"], "k--", lw=1, label="target")
    ax.set_ylabel("normal force [N]"); ax.legend(fontsize=8, loc="upper left")
    ax2 = fig.add_subplot(gs[2, :], sharex=ax)
    n_true = pn.normal(L["tip"][:, 0], L["tip"][:, 1])
    m = np.isin(L["phase"], ["approach", "sweep", "dwell"])
    ax2.plot(t[m], np.degrees(np.arccos(np.clip(np.sum(L["axis"][m] * -n_true[m], 1), -1, 1))), ".", ms=1,
             color="#8250df", label="tool axis vs true normal")
    slope = np.degrees(np.arccos(n_true[m, 2]))
    ax2.plot(t[m], slope, color="0.6", lw=0.8, label="surface slope under the tip")
    ax2.set_ylabel("deg"); ax2.set_xlabel("time [s]"); ax2.legend(fontsize=8, loc="upper left")
    for a in (ax, ax2):
        for t0, nm in task.phases:
            a.axvline(t0, color="0.85", lw=0.8)
        a.grid(alpha=0.3)
    for t0, nm in task.phases:
        ax.text(t0 + 0.05, 0.92, nm, fontsize=7, color="0.4", transform=ax.get_xaxis_transform())
    fig.suptitle(f"Force-controlled tracing of a random freeform surface — {args.robot}, {args.force:g} N, "
                 f"seed {args.panel_seed}, {args.perception}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="go2_z1")
    ap.add_argument("--world", choices=list(WORLDS), default="workshop")
    ap.add_argument("--panel-seed", type=int, default=0)
    ap.add_argument("--panel-amp", type=float, default=0.03, help="max height deviation of the random surface [m]")
    ap.add_argument("--perception", choices=["wrist", "flat"], default="wrist")
    ap.add_argument("--force", type=float, default=5.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.03)
    ap.add_argument("--force-noise", type=float, default=0.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = ROBOTS[args.robot]()
    world = WORLDS[args.world](panel_seed=args.panel_seed, panel_amp=args.panel_amp)
    out = os.path.join(ROOT, "results", "surface")
    os.makedirs(out, exist_ok=True)
    L, task, wbc, plant, solve_ms, frames, info = run(args, cfg, world, render=not args.no_video)
    res = metrics(L, task, wbc, plant, solve_ms, world.panels[0], args, info)
    print(json.dumps(res, indent=1, default=float))
    name = f"{args.robot}_seed{args.panel_seed}_{args.perception}{args.tag}"
    plot(L, task, world.panels[0], args, os.path.join(out, f"{name}.png"))
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    if frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"{name}.mp4"), frames, fps=50, quality=7, macro_block_size=1)


if __name__ == "__main__":
    main()
