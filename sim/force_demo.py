"""Force control demo: the robot presses the torch tip on the curved pipe with a constant normal force while sweeping
an arc over the pipe top (see sim/force_task.py). The real pipe can differ from the map; the robot can sense it.

    python sim/force_demo.py                                      # Go2 + Z1, 5 N, +-30 deg, pipe as in the map
    python sim/force_demo.py --pipe-dy 0.06 --pipe-dyaw 6 --pipe-radius 0.12 --perception vision --normal touch
    python sim/force_demo.py --robot anymal_c_z1 --force 10 --passes 4
    python sim/force_demo.py --ctrl-torch-mass 0.6 --force-noise 0.3

--perception map     the robot believes the map's pipe (position, direction, radius)
--perception vision  head depth camera scan from the inspection pose (beside the map's docking pose, see
                     world.pipe_inspect_pose), cylinder fit (sim/perception/), dock and plan on the *measured* pipe
--normal model       surface normal / arc from the believed pipe
--normal touch       once in contact, the normal comes from the contact-force direction (tactile surface following)

The robot is placed at the inspection / docking poses directly (sim/mission_demo.py walks there). All accuracy
metrics are measured against the real pipe.
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from force_task import PipeForceTask, tip_contact_force  # noqa: E402
from perception.cylinder_fit import check_estimate, fit_cylinder  # noqa: E402
from perception.depth_camera import DepthCamera, scan  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import PIPE_WORK_X, WORLDS, pipe_dock_pose, pipe_frame, pipe_inspect_pose  # noqa: E402

DT = 0.002


class ForceSensor:
    """Contact-tip force 'sensor': MuJoCo contact force (magnitude + direction) + optional Gaussian noise, first-order
    low-pass filtered."""

    def __init__(self, m, noise=0.0, tau=0.005, seed=0):
        self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "contact_tip")
        self.noise, self.alpha = noise, DT / (tau + DT)
        self.rng = np.random.default_rng(seed)
        self.f, self.raw = 0.0, 0.0
        self.vec = np.zeros(3)

    def read(self, m, d):
        self.raw, _, vec = tip_contact_force(m, d, self.gid)
        if self.noise > 0:
            vec = vec + self.rng.normal(0, self.noise, 3)
        self.vec += self.alpha * (vec - self.vec)
        self.f = float(np.linalg.norm(self.vec))
        return self.f


class Recorder:
    def __init__(self, plant, cam):
        self.renderer = mujoco.Renderer(plant, 480, 854)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = cam["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        self.frames = []

    def frame(self, d, text, tip=None, n=None, force=0.0, cloud=None):
        self.renderer.update_scene(d, camera=self.cam)
        scn = self.renderer.scene
        if tip is not None and force > 0.05 and scn.ngeom < scn.maxgeom:
            # contact-force arrow (2 cm per N) pointing into the surface
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.1, 0.8, 1.0, 1], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, tip + (0.02 * force + 0.01) * n, tip + 0.01 * n)
            scn.ngeom += 1
        if cloud is not None:
            draw_cloud(scn, *cloud)
        from PIL import Image, ImageDraw
        im = Image.fromarray(self.renderer.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 38], fill=(0, 0, 0))
        dr.multiline_text((6, 3), text, fill=(255, 255, 255))
        self.frames.append(np.asarray(im))


def draw_cloud(scn, pts, rgba, size=0.004, max_pts=1500):
    for p in pts[:: max(1, len(pts) // max_pts)]:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0], p, np.eye(3).ravel(),
                            np.array(rgba, np.float32))
        scn.ngeom += 1


def perceive_pipe(m, d, ix, cfg, map_pipe, frames=5):
    """Stand at the inspection pose (map-based) and fit a cylinder to the head depth camera's point cloud."""
    insp = pipe_inspect_pose(map_pipe, cfg.pipe_work_dist)
    initial_state(m, d, ix, xy=insp[:2], yaw=insp[2])
    cam = DepthCamera(m)
    t0 = time.perf_counter()
    pts, c = scan(m, d, cam, frames)
    est = check_estimate(fit_cylinder(pts, c, prior=map_pipe), map_pipe)
    est.extra["fit_ms"] = 1000 * (time.perf_counter() - t0)
    return est


def pipe_errors(est, true):
    v = est.end - true.end
    return dict(radius_err_mm=1000 * (est.radius - true.radius),
                axis_err_deg=float(np.degrees(np.arccos(min(1.0, abs(est.axis @ true.axis))))),
                axis_offset_mm=float(1000 * np.linalg.norm(v - (v @ true.axis) * true.axis)),
                end_err_mm=float(1000 * (v @ true.axis)))


def run(args, cfg, world, render=True, viewer=False):
    true_pipe, map_pipe = world.pipes[0], world.map_pipes[0]
    plant = build_model(cfg, torch_mass=args.torch_mass, world=world, contact_tip=True)
    ctrl = controller_model(cfg, args.ctrl_torch_mass)
    plant.opt.timestep = ctrl.opt.timestep = DT
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    info = dict(perception=args.perception, normal=args.normal)
    if args.perception == "vision":
        belief = perceive_pipe(plant, d, ix, cfg, map_pipe)
        info.update(estimate={k: round(float(v), 3) for k, v in pipe_errors(belief, true_pipe).items()},
                    fit_ms=belief.extra["fit_ms"], cloud_inliers=belief.inliers)
    else:
        belief = map_pipe
    info["belief_vs_true"] = {k: round(float(v), 3) for k, v in pipe_errors(belief, true_pipe).items()}
    dock = pipe_dock_pose(belief, cfg.pipe_work_dist)
    initial_state(plant, d, ix, xy=dock[:2], yaw=dock[2])
    task = PipeForceTask(cfg, ctrl, d.qpos.copy(), belief, f_des=args.force, theta_max=np.radians(args.arc_deg),
                         passes=args.passes, speed=args.speed, normal_source=args.normal)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    wp = true_pipe.surface_point(PIPE_WORK_X, 0.0)[0]
    rec = Recorder(plant, dict(lookat=[*(wp[:2] - 0.2 * true_pipe.axis[:2]), wp[2] + 0.1], distance=1.5,
                               azimuth=np.degrees(true_pipe.yaw) + 200, elevation=-22)) if render else None
    v = None
    if viewer:
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch_passive(plant, d)

    L = {k: [] for k in ["t", "f", "f_raw", "f_des", "phase", "tip", "tip_ref", "n", "axis", "base_pos", "base_quat",
                         "feet", "tau", "vn_des"]}
    solve_ms = []
    for k in range(int(task.T / DT)):
        t = d.time
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas, sensor.vec.copy())
        c0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - c0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        log_step(L, d, ix, task, ref, sensor, wbc, t, t)
        if v is not None:
            if not v.is_running():
                break
            v.sync()
            time.sleep(max(0.0, DT - 1e-4))
        if rec is not None and k % 10 == 9:
            ph = L["phase"][-1]
            base_dev = 1000 * np.linalg.norm(d.qpos[0:3] - L["base_pos"][int(task.start["unstow"] / DT)]) if t > task.start["unstow"] else 0.0
            th = np.degrees(true_theta(d.site_xpos[ix.tip], true_pipe))
            txt = (f"force control  {cfg.name}  [{args.perception} / {args.normal}]  t={t:5.2f}s  [{ph}]\n"
                   f"F = {f_meas:5.2f} N (target {args.force:.1f})   arc on the real pipe {th:+5.1f} deg   "
                   f"base drift {base_dev:5.2f} mm")
            cloud = (belief.extra["inlier_points"], [0.1, 1.0, 0.3, 1]) if args.perception == "vision" and ph in ("settle", "brace") else None
            rec.frame(d, txt, d.site_xpos[ix.tip].copy(), L["n"][-1], f_meas, cloud)
    if v is not None:
        v.close()
    L = {k: (np.array(x) if k != "phase" else np.array(x, dtype=object)) for k, x in L.items()}
    return L, task, wbc, plant, solve_ms, (rec.frames if rec else []), info


def log_step(L, d, ix, task, ref, sensor, wbc, t, t_rel):
    ph = task.phase(t)
    n = ref["force"]["n"] if "force" in ref else task.pipe.surface_point(task.s_work, task.theta(t_rel))[1]
    L["t"].append(t_rel); L["f"].append(sensor.f); L["f_raw"].append(sensor.raw); L["phase"].append(ph)
    L["f_des"].append(ref["force"]["f_des"] if "force" in ref else np.nan)
    L["tip"].append(d.site_xpos[ix.tip].copy()); L["tip_ref"].append(ref.get("ee_pos", np.full(3, np.nan)))
    L["n"].append(n); L["axis"].append(d.site_xmat[ix.tip].reshape(3, 3)[:, 0].copy())
    L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
    L["feet"].append(d.geom_xpos[ix.feet].copy()); L["tau"].append(d.actuator_force.copy())
    L["vn_des"].append(wbc.vn_des if "force" in ref else np.nan)


def true_normal(tip, pipe):
    v = tip - pipe.end
    radial = v - (v @ pipe.axis) * pipe.axis
    return radial / (np.linalg.norm(radial) + 1e-12)


def true_theta(tip, pipe):
    n = true_normal(tip, pipe)
    up, side = pipe_frame(pipe.axis)
    return float(np.arctan2(n @ side, n @ up))


def rot_dev_deg(q_ref, Q):
    out = []
    for q in Q:
        e = np.zeros(3); mujoco.mju_subQuat(e, q, q_ref); out.append(np.degrees(np.linalg.norm(e)))
    return np.array(out)


def metrics(L, task, wbc, plant, solve_ms, args, true_pipe):
    t, ph = L["t"], L["phase"]
    sweep = ph == "sweep"
    contact = np.isin(ph, ["sweep", "dwell"])
    i0 = int(task.start["unstow"] / DT)
    after = t >= task.start["unstow"]
    f, fe = L["f_raw"], L["f_raw"] - args.force
    # accuracy against the real pipe
    n_true = np.array([true_normal(p, true_pipe) for p in L["tip"]])
    th_true = np.degrees([true_theta(p, true_pipe) for p in L["tip"][sweep]])
    axis_e = np.degrees(np.arccos(np.clip(np.sum(L["axis"] * -n_true, 1), -1, 1)))
    s_true = (L["tip"] - true_pipe.end) @ true_pipe.axis
    app = ph == "approach"
    t_contact = t[app][np.argmax(f[app] > 0.05)] - task.start["approach"] if np.any(f[app] > 0.05) else np.nan
    pos_dev = 1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1)
    slip = 1000 * np.max(np.linalg.norm((L["feet"] - L["feet"][i0])[:, :, :2], axis=2), axis=1)
    return dict(
        robot=args.robot, f_des=args.force, arc_deg=args.arc_deg, passes=args.passes, speed_mm_s=1000 * args.speed,
        torch_mass=args.torch_mass, ctrl_torch_mass=args.ctrl_torch_mass, force_noise=args.force_noise,
        sweep_f_mean=float(f[sweep].mean()), sweep_f_rms_err=float(np.sqrt(np.mean(fe[sweep] ** 2))),
        sweep_f_min=float(f[sweep].min()), sweep_f_max=float(f[sweep].max()),
        contact_lost_frac=float(np.mean(f[contact] < 0.05)),
        approach_contact_s=float(t_contact), approach_f_peak=float(f[app].max()),
        arc_on_real_pipe_deg=[round(float(th_true.min()), 1), round(float(th_true.max()), 1)],
        axis_err_vs_real_normal_rms_deg=float(np.sqrt(np.mean(axis_e[sweep] ** 2))),
        axis_err_vs_real_normal_max_deg=float(axis_e[sweep].max()),
        axial_drift_rms_mm=float(1000 * np.sqrt(np.mean((s_true[sweep] - np.mean(s_true[sweep])) ** 2))),
        base_pos_dev_max_mm=float(pos_dev[after].max()),
        base_rot_dev_max_deg=float(rot_dev_deg(L["base_quat"][i0], L["base_quat"])[after].max()),
        foot_slip_max_mm=float(slip[after].max()),
        peak_torque_frac=float(np.max(np.abs(L["tau"][after]) / plant.actuator_forcerange[:, 1])),
        arc_ik_residual_mm=float(task.arc_ik_residual_mm),
        qp_ms_mean=float(np.mean(solve_ms)), qp_ms_max=float(np.max(solve_ms)), qp_failures=wbc.fail,
        fell=bool(np.any(L["base_pos"][:, 2] < 0.15)))


def plot(L, task, args, true_pipe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = L["t"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, L["f_raw"], color="#cf222e", lw=0.8, label="measured (contact)")
    ax[0].plot(t, L["f_des"], "k--", lw=1, label="target")
    ax[0].set_ylabel("normal force [N]"); ax[0].legend(fontsize=8, loc="upper left")
    m = np.isin(L["phase"], ["approach", "sweep", "dwell"])
    ax[1].plot(t, np.degrees([task.theta(x) for x in t]), color="0.6", lw=1, label="planned")
    ax[1].plot(t[m], np.degrees([true_theta(p, true_pipe) for p in L["tip"][m]]), color="#1f6feb", lw=1.2,
               label="on the real pipe")
    ax[1].set_ylabel("arc angle [deg]"); ax[1].legend(fontsize=8, loc="upper left")
    n_true = np.array([true_normal(p, true_pipe) for p in L["tip"][m]])
    axis_e = np.degrees(np.arccos(np.clip(np.sum(L["axis"][m] * -n_true, 1), -1, 1)))
    ax[2].plot(t[m], axis_e, ".", ms=1, color="#8250df")
    ax[2].set_ylabel("tool axis vs real\nsurface normal [deg]")
    i0 = int(task.start["unstow"] / DT)
    a3 = t >= task.start["unstow"]  # after the (intentional) brace
    ax[3].plot(t[a3], 1000 * np.linalg.norm(L["base_pos"][a3] - L["base_pos"][i0], axis=1), color="#1f6feb")
    ax[3].set_ylabel("base drift since unstow [mm]"); ax[3].set_xlabel("time [s]")
    for a in ax:
        for t0, nm in task.phases:
            a.axvline(t0, color="0.8", lw=0.8)
        a.grid(alpha=0.3)
    for t0, nm in task.phases:
        ax[0].text(t0 + 0.05, 0.92, nm, fontsize=7, color="0.4", transform=ax[0].get_xaxis_transform())
    fig.suptitle(f"Force control on a curved pipe — {args.robot}, target {args.force:g} N, arc ±{args.arc_deg:g}°, "
                 f"{args.perception} / {args.normal}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="go2_z1")
    ap.add_argument("--world", choices=list(WORLDS), default="workshop")
    ap.add_argument("--force", type=float, default=5.0, help="target normal force [N]")
    ap.add_argument("--arc-deg", type=float, default=30.0, help="sweep +- this angle over the pipe top [deg]")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.03, help="sweep speed along the surface [m/s]")
    ap.add_argument("--torch-mass", type=float, default=1.2)
    ap.add_argument("--ctrl-torch-mass", type=float, default=None)
    ap.add_argument("--force-noise", type=float, default=0.0, help="force sensor noise std per axis [N]")
    ap.add_argument("--perception", choices=["map", "vision"], default="vision")
    ap.add_argument("--normal", choices=["model", "touch"], default="model")
    ap.add_argument("--pipe-dx", type=float, default=0.0, help="real pipe vs map: shift along x [m]")
    ap.add_argument("--pipe-dy", type=float, default=0.0, help="real pipe vs map: shift along y [m]")
    ap.add_argument("--pipe-dyaw", type=float, default=0.0, help="real pipe vs map: rotation [deg]")
    ap.add_argument("--pipe-radius", type=float, default=None, help="real pipe radius [m] (map: 0.15)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.ctrl_torch_mass = args.torch_mass if args.ctrl_torch_mass is None else args.ctrl_torch_mass
    cfg = ROBOTS[args.robot]()
    world = WORLDS[args.world](pipe_dx=args.pipe_dx, pipe_dy=args.pipe_dy, pipe_dyaw_deg=args.pipe_dyaw,
                               pipe_radius=args.pipe_radius)
    out = os.path.join(ROOT, "results", "force_control")
    os.makedirs(out, exist_ok=True)
    L, task, wbc, plant, solve_ms, frames, info = run(args, cfg, world, render=not (args.no_video or args.viewer),
                                                      viewer=args.viewer)
    res = {**info, **metrics(L, task, wbc, plant, solve_ms, args, world.pipes[0])}
    print(json.dumps(res, indent=1))
    name = f"{args.robot}_{args.perception}_{args.normal}{args.tag}"
    plot(L, task, args, world.pipes[0], os.path.join(out, f"{name}.png"))
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    if frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"{name}.mp4"), frames, fps=50, quality=7, macro_block_size=1)


if __name__ == "__main__":
    main()
