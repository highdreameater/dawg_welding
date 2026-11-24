"""Full mission in the workshop: plan -> walk (RL) -> look (depth camera) -> dock -> force-controlled pass -> walk home.

    start --A* on the map--> inspection pose (beside where the map says the docking pose is)
        -> stand, depth-camera scan, cylinder fit: the pipe's real position, direction and radius
        -> walk the straight docking leg to the pose computed from the *measured* pipe
        -> hand-off: stance WBC braces, unstows, presses 5 N while sweeping +-30 deg (touch-guided), stows
        -> hand-back: RL policy, planned path back to the start

    python sim/mission_demo.py                                  # Go2 + Z1, payload-trained policy, pipe moved vs map
    python sim/mission_demo.py --loco go2                       # pretrained Isaac Lab Go2 policy instead
    python sim/mission_demo.py --perception map --pipe-dy 0     # no sensing, pipe exactly as in the map
    python sim/mission_demo.py --robot anymal_c_z1 --no-video
    python sim/mission_demo.py --station panel                  # random freeform surface: dock, wrist scan, trace
    python sim/mission_demo.py --station panel --panel-seed 3 --panel-amp 0.04

Locomotion: sim/locomotion/policy.py (Isaac Lab velocity policies in MuJoCo; "go2_payload" = fine-tuned with the arm
by sim/locomotion/train_payload.py). The map is known (sim/world.py) - no SLAM; the real pipe may differ from it.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace  # noqa: F401

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from force_demo import ForceSensor, draw_cloud, log_step, metrics as force_metrics, pipe_errors  # noqa: E402
from force_task import PipeForceTask  # noqa: E402
from locomotion.policy import LOCO, ROBOT_LOCO, SIM_DT as RL_DT, VelocityPolicy  # noqa: E402
from perception.cylinder_fit import check_estimate, fit_cylinder  # noqa: E402
from perception.depth_camera import DepthCamera  # noqa: E402
from perception.surface_fit import fit_surface  # noqa: E402
from surface_demo import WRIST_CAM, metrics as surface_metrics  # noqa: E402
from surface_task import SurfaceTraceTask  # noqa: E402
from planning.follower import PathFollower, wrap  # noqa: E402
from planning.grid_planner import OccupancyGrid, plan_path, resample  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import PIPE_WORK_X, WORLDS, panel_dock_pose, pipe_dock_pose, pipe_inspect_pose  # noqa: E402

WBC_DT = 0.002
FPS = 25


def base_pose(d, ix):
    R = d.xmat[ix.base].reshape(3, 3)
    return d.qpos[0:2].copy(), float(np.arctan2(R[1, 0], R[0, 0]))


class Video:
    def __init__(self, m, ix, work_cam):
        self.r = mujoco.Renderer(m, 480, 854)
        self.track = mujoco.MjvCamera()
        self.track.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.track.trackbodyid = ix.base
        self.track.distance, self.track.azimuth, self.track.elevation = 3.0, 225, -50
        self.work = mujoco.MjvCamera()
        self.work.lookat[:] = work_cam["lookat"]
        self.work.distance, self.work.azimuth, self.work.elevation = work_cam["distance"], work_cam["azimuth"], work_cam["elevation"]
        self.frames, self.markers, self.cloud, self.last = [], [], None, -1.0

    def due(self, t):
        return t - self.last >= 1.0 / FPS - 1e-9

    def frame(self, d, text, cam="track"):
        from PIL import Image, ImageDraw
        self.last = d.time
        self.r.update_scene(d, camera=self.track if cam == "track" else self.work)
        scn = self.r.scene
        for p, rgba in self.markers:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], p,
                                np.eye(3).ravel(), np.array(rgba, np.float32))
            scn.ngeom += 1
        if self.cloud is not None:
            draw_cloud(scn, *self.cloud)
        im = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 38], fill=(0, 0, 0))
        dr.multiline_text((6, 3), text, fill=(255, 255, 255))
        self.frames.append(np.asarray(im))


def world_collisions(m, d, foot_geoms):
    """Robot geoms (other than the feet) touching the world (walls, obstacles, pipe)."""
    hits = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 == 0) == (b2 == 0):
            continue
        g_robot = c.geom2 if b1 == 0 else c.geom1
        g_world = c.geom1 if b1 == 0 else c.geom2
        if g_robot in foot_geoms or m.geom_type[g_world] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        hits.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_world) or str(g_world))
    return hits


def walk(m, d, ix, pol, follower, grid, video, log, label, t_max=60.0, stand=1.0, on_stand=None):
    """Follow the path with the RL policy until docked (or timeout), then stand still for `stand` seconds.
    on_stand(d) is called every policy step while standing (e.g. to grab camera frames)."""
    m.opt.timestep = RL_DT
    foot_geoms = set(ix.feet.tolist())
    t0, t_near, t_done = d.time, None, None
    while d.time - t0 < t_max:
        xy, yaw = base_pose(d, ix)
        cmd = follower.command(xy, yaw) if t_done is None else np.zeros(3)
        if t_done is None:
            # the policy cannot servo to millimetres: accept a dock that stays within 5 cm / 4 deg for 1 s
            near = np.linalg.norm(follower.goal - xy) < 0.05 and abs(wrap(follower.goal_yaw - yaw)) < np.radians(4)
            t_near = (t_near or d.time) if near else None
            if follower.done or (t_near is not None and d.time - t_near > 1.0):
                follower.done, t_done = True, d.time
        elif d.time - t_done > stand:
            break
        elif on_stand is not None and pol.k % pol.decimation == 0 and d.time - t_done > 0.5:
            on_stand(d)
        pol.torques(m, d, cmd)
        mujoco.mj_step(m, d)
        hits = world_collisions(m, d, foot_geoms)
        R = d.xmat[ix.base].reshape(3, 3)
        log.append(dict(t=d.time, mode=label, x=xy[0], y=xy[1], yaw=yaw, cmd=cmd.copy(), z=d.qpos[2],
                        tilt=float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
                        v=(R.T @ d.qvel[0:3]).copy(), wz=float(d.qvel[5]),
                        clearance=grid.clearance_at(xy), hits=sorted(hits)))
        if d.qpos[2] < 0.12 or R[2, 2] < 0.5:
            raise RuntimeError(f"robot fell while walking ({label}) at t={d.time:.2f}s")
        if video is not None and video.due(d.time):
            video.frame(d, f"MISSION  [{label}: {'walking' if t_done is None else 'standing'}]  t={d.time:5.1f}s\n"
                           f"cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}   "
                           f"to goal {np.linalg.norm(follower.goal - xy):.2f} m")
    xy, yaw = base_pose(d, ix)
    return dict(time_s=d.time - t0, docked=bool(follower.done),
                pos_err_mm=1000 * float(np.linalg.norm(follower.goal - xy)),
                yaw_err_deg=float(np.degrees(abs(wrap(follower.goal_yaw - yaw)))))


def force_phase(plant, ctrl, d, ix, cfg, belief, true_pipe, args, video):
    plant.opt.timestep = WBC_DT
    task = PipeForceTask(cfg, ctrl, d.qpos.copy(), belief, f_des=args.force, theta_max=np.radians(args.arc_deg),
                         passes=args.passes, speed=args.speed, t0=d.time, normal_source=args.normal)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    L = {k: [] for k in ["t", "f", "f_raw", "f_des", "phase", "tip", "tip_ref", "n", "axis", "base_pos", "base_quat",
                         "feet", "tau", "vn_des"]}
    solve_ms = []
    for _ in range(int(round(task.T / WBC_DT))):
        t = d.time
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas, sensor.vec.copy())
        c0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - c0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        log_step(L, d, ix, task, ref, sensor, wbc, t, t - task.t0)
        if video is not None and video.due(d.time):
            video.frame(d, f"MISSION  [stance WBC + force control: {L['phase'][-1]}]  t={d.time:5.1f}s\n"
                           f"F = {f_meas:5.2f} N (target {args.force:.1f})   planned arc "
                           f"{np.degrees(task.theta(t - task.t0)):+5.1f} deg   ({args.normal}-guided)", cam="work")
    L = {k: (np.array(x) if k != "phase" else np.array(x, dtype=object)) for k, x in L.items()}
    return force_metrics(L, task, wbc, plant, solve_ms, args, true_pipe)


def surface_phase(plant, ctrl, d, ix, cfg, panel, args, video):
    """Stance WBC at the panel: brace, wrist-camera scan, reconstruct, force-controlled S-path, stow."""
    plant.opt.timestep = WBC_DT
    prior = (panel.cx, panel.cy, panel.hx, panel.hy, panel.top_z)   # the map: stand footprint + nominal height
    task = SurfaceTraceTask(cfg, ctrl, d.qpos.copy(), prior, f_des=args.force, passes=args.passes, speed=args.speed,
                            t0=d.time)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    cam = DepthCamera(plant, WRIST_CAM)
    clouds, info, marks = [], {}, None
    L = {k: [] for k in ["t", "f_raw", "f_des", "phase", "tip", "axis", "n_cmd", "u", "base_pos", "base_quat", "arm_q",
                         "feet", "tau"]}
    solve_ms, k = [], 0
    while d.time < task.t0 + task.T:
        t = d.time
        ph = task.phase(t)
        if ph == "scan" and t - task.t0 > task.start["scan"] + 0.4 and k % 10 == 0:
            clouds.append(cam.capture(plant, d)[0])
        if task.model is None and ph not in ("settle", "brace", "to_scan", "scan"):
            c0 = time.perf_counter()
            model = fit_surface(np.concatenate(clouds), prior[:4])
            task.set_model(model)
            info = dict(perception="wrist", frames=len(clouds), raw_points=int(sum(len(c) for c in clouds)),
                        model=dict(model.info), model_ms=1000 * (time.perf_counter() - c0), path=task.path_choice)
            marks = np.array([task.path_point(u)[0] for u in np.linspace(0, 1, 60)]) + [0, 0, 0.002]
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas, sensor.vec.copy())
        c0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - c0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        L["t"].append(t - task.t0); L["f_raw"].append(sensor.raw); L["phase"].append(task.phase(t))
        L["f_des"].append(ref["force"]["f_des"] if "force" in ref else np.nan)
        L["tip"].append(d.site_xpos[ix.tip].copy()); L["axis"].append(d.site_xmat[ix.tip].reshape(3, 3)[:, 0].copy())
        L["n_cmd"].append(ref["force"]["n"] if "force" in ref else np.full(3, np.nan)); L["u"].append(task.u(t - task.t0))
        L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
        L["arm_q"].append(d.qpos[ix.arm_q].copy())
        L["feet"].append(d.geom_xpos[ix.feet].copy()); L["tau"].append(d.actuator_force.copy())
        if video is not None and video.due(d.time):
            p_ = L["phase"][-1]
            video.cloud = ((clouds[-1][::3], [0.1, 1.0, 0.3, 1]) if clouds and p_ in ("scan", "to_hover")
                           else (marks, [1.0, 0.85, 0.1, 1]) if marks is not None and p_ in ("hover", "approach", "sweep", "dwell")
                           else None)
            video.frame(d, f"MISSION  [stance WBC at the freeform panel: {p_}]  t={d.time:5.1f}s\n"
                           f"F = {sensor.raw:5.2f} N (target {args.force:.1f})   wrist-camera scan -> thin-plate "
                           f"spline -> force-controlled S-path", cam="work")
        k += 1
    if video is not None:
        video.cloud = None
    L = {k_: (np.array(x) if k_ != "phase" else np.array(x, dtype=object)) for k_, x in L.items()}
    return surface_metrics(L, task, wbc, plant, solve_ms, panel, args, info)


def plot_map(world, grid, paths, log, path_png, est=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.imshow(np.where(grid.occ, 2, np.where(grid.free, 0, 1)).T, origin="lower", extent=list(world.bounds),
              cmap="Greys", vmax=2.6)
    for (label, p, info), col in zip(paths, ["#1f6feb", "#2da44e", "#8250df"]):
        if info is not None:
            ax.plot(info["raw"][:, 0], info["raw"][:, 1], ":", color=col, lw=1)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2, alpha=0.5, label=f"planned ({label})")
    for label, col in [("to_inspect", "#cf222e"), ("dock", "#bf3989"), ("home", "#bf8700")]:
        tr = np.array([(r["x"], r["y"]) for r in log if r["mode"] == label])
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.2, label=f"executed ({label})")

    def outline(p, **kw):
        a, n = p.axis[:2] / np.linalg.norm(p.axis[:2]), np.array([-p.axis[1], p.axis[0]])
        L = getattr(p, "length", None) or p.length_seen
        e = p.end[:2]
        q = np.array([e + p.radius * n, e + L * a + p.radius * n, e + L * a - p.radius * n, e - p.radius * n,
                      e + p.radius * n])
        ax.plot(q[:, 0], q[:, 1], **kw)

    outline(world.map_pipes[0], color="#1f6feb", lw=1, ls="--", label="pipe in the map")
    outline(world.pipes[0], color="k", lw=1.5, label="real pipe")
    if est is not None:
        outline(est, color="#2da44e", lw=1.5, ls="-.", label="pipe measured (depth camera)")
    ax.plot(*world.start[:2], "ko", ms=7)
    ax.annotate(" start", world.start[:2], fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Workshop map (black: obstacles, grey: robot-radius margin); dotted: raw A*")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path_png, dpi=100)
    x0, y0 = world.pipes[0].end[:2]
    ax.set_xlim(x0 - 1.6, x0 + 0.6); ax.set_ylim(y0 - 0.8, y0 + 0.8)
    ax.set_title("Pipe station: map vs real vs measured")
    fig.savefig(path_png.replace(".png", "_station.png"), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="go2_z1")
    ap.add_argument("--loco", default="auto", help="locomotion policy (sim/locomotion/policy.py LOCO); auto: "
                                                   "go2_payload if trained, else the robot's pretrained policy")
    ap.add_argument("--world", choices=list(WORLDS), default="workshop")
    ap.add_argument("--station", choices=["pipe", "panel"], default="pipe",
                    help="pipe: curved pipe found by the head camera; panel: random freeform surface, wrist scan")
    ap.add_argument("--panel-seed", type=int, default=4)
    ap.add_argument("--panel-amp", type=float, default=0.03)
    ap.add_argument("--perception", choices=["map", "vision"], default="vision")
    ap.add_argument("--normal", choices=["model", "touch"], default="model")
    ap.add_argument("--pipe-dx", type=float, default=0.04)
    ap.add_argument("--pipe-dy", type=float, default=0.06)
    ap.add_argument("--pipe-dyaw", type=float, default=6.0)
    ap.add_argument("--pipe-radius", type=float, default=0.12)
    ap.add_argument("--force", type=float, default=5.0)
    ap.add_argument("--arc-deg", type=float, default=30.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.03)
    ap.add_argument("--force-noise", type=float, default=0.0)
    ap.add_argument("--no-return", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = ROBOTS[args.robot]()
    world = WORLDS[args.world](pipe_dx=args.pipe_dx, pipe_dy=args.pipe_dy, pipe_dyaw_deg=args.pipe_dyaw,
                               pipe_radius=args.pipe_radius, panel_seed=args.panel_seed, panel_amp=args.panel_amp)
    args.torch_mass = args.ctrl_torch_mass = 1.2
    if args.loco == "auto":
        args.loco = ROBOT_LOCO[args.robot]
        if args.robot == "go2_z1" and os.path.exists(LOCO["go2_payload"].checkpoint):
            args.loco = "go2_payload"
    map_pipe, true_pipe = world.map_pipes[0], world.pipes[0]
    out = os.path.join(ROOT, "results", "mission")
    os.makedirs(out, exist_ok=True)

    plant = build_model(cfg, world=world, contact_tip=True)
    ctrl = controller_model(cfg)
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    initial_state(plant, d, ix, xy=world.start[:2], yaw=world.start[2])
    pol = VelocityPolicy(plant, cfg, loco=args.loco)
    grid = OccupancyGrid(world)
    panel = world.panels[0]
    if args.station == "panel":
        work_cam = dict(lookat=[panel.cx - 0.12, panel.cy, panel.top_z + 0.1], distance=1.3, azimuth=215, elevation=-32)
    else:
        wp = true_pipe.surface_point(PIPE_WORK_X, 0.0)[0]
        work_cam = dict(lookat=[*(wp[:2] - 0.2 * true_pipe.axis[:2]), wp[2] + 0.1], distance=1.5,
                        azimuth=np.degrees(true_pipe.yaw) + 200, elevation=-22)
    video = None if args.no_video else Video(plant, ix, work_cam)
    log, paths = [], []
    summary = dict(robot=args.robot, loco=args.loco, station=args.station)
    if args.station == "panel":
        summary.update(panel_seed=args.panel_seed, panel_amp=args.panel_amp)
    else:
        summary.update(perception=args.perception, normal=args.normal,
                       pipe_vs_map=dict(dx=args.pipe_dx, dy=args.pipe_dy, dyaw_deg=args.pipe_dyaw,
                                        radius=true_pipe.radius))

    # 1) plan on the map to the inspection pose (or straight to the map's docking pose without perception)
    if args.station == "panel":   # the panel's shape is sensed at the station (wrist camera): dock directly
        args.perception = "map"
        dock_map = panel_dock_pose(panel, cfg.pipe_work_dist)
    else:
        dock_map = pipe_dock_pose(map_pipe, cfg.pipe_work_dist)
    goal = pipe_inspect_pose(map_pipe, cfg.pipe_work_dist) if args.perception == "vision" else dock_map
    c0 = time.perf_counter()
    path, info = plan_path(grid, world.start[:2], goal[:2], goal[2])
    summary["plan_out"] = dict(plan_ms=1000 * (time.perf_counter() - c0), length_m=info["length"],
                               min_clearance_m=info["min_clearance"])
    paths.append(("out", path, info))
    if video:
        video.markers = [(np.array([*p, 0.02]), [0.1, 0.5, 1.0, 1]) for p in path[::4]]

    # 2) walk; with perception: stand at the inspection pose and grab depth frames from the head camera
    cam, clouds = DepthCamera(plant), []
    grab = (lambda dd: clouds.append(cam.capture(plant, dd)[:2])) if args.perception == "vision" else None
    leg = "to_inspect" if args.perception == "vision" else "dock"
    summary["walk_out"] = walk(plant, d, ix, pol, PathFollower(path, goal[2]), grid, video, log, leg,
                               stand=1.0 if grab is None else 0.7, on_stand=grab)
    est = None
    if args.perception == "vision":
        # 3) cylinder fit on the stacked frames -> docking pose from the *measured* pipe
        if not clouds:
            raise RuntimeError(f"never settled at the inspection pose: {summary['walk_out']}")
        c0 = time.perf_counter()
        pts = np.concatenate([c[0] for c in clouds])
        est = check_estimate(fit_cylinder(pts, clouds[-1][1], prior=map_pipe), map_pipe)
        summary["perception"] = dict(frames=len(clouds), points=int(len(pts)), fit_ms=1000 * (time.perf_counter() - c0),
                                     inliers=est.inliers, measured_radius_mm=1000 * est.radius,
                                     **{k: float(v) for k, v in pipe_errors(est, true_pipe).items()},
                                     map_vs_real={k: float(v) for k, v in pipe_errors(map_pipe, true_pipe).items()})
        dock = pipe_dock_pose(est, cfg.pipe_work_dist)
        xy, _ = base_pose(d, ix)
        line = resample(np.array([xy, dock[:2]]), 0.05)
        paths.append(("dock (from scan)", line, None))
        if video:
            video.markers = []
            video.cloud = (est.extra["inlier_points"], [0.1, 1.0, 0.3, 1])
            for _ in range(int(1.5 * FPS)):  # hold the view on the measured cylinder points
                video.frame(d, f"MISSION  [depth camera: cylinder fit]  r = {1000 * est.radius:.1f} mm "
                               f"(map {1000 * map_pipe.radius:.0f})\npipe moved vs map by "
                               f"{1000 * pipe_errors(map_pipe, true_pipe)['axis_offset_mm'] / 1000:.0f} mm, "
                               f"{pipe_errors(map_pipe, true_pipe)['axis_err_deg']:.1f} deg - docking on the measured pipe")
            video.cloud = None
        summary["walk_dock"] = walk(plant, d, ix, pol, PathFollower(line, dock[2], dock_radius=1.2), grid, video,
                                    log, "dock")
        belief = est
    else:
        belief = map_pipe

    # 4) hand-off to the stance WBC: force-controlled pass over the pipe / the freeform panel
    if video:
        video.markers = []
    if args.station == "panel":
        summary["surface"] = surface_phase(plant, ctrl, d, ix, cfg, panel, args, video)
    else:
        summary["force"] = force_phase(plant, ctrl, d, ix, cfg, belief, true_pipe, args, video)

    # 5) hand back to the RL policy and walk home
    if not args.no_return:
        pol.reset()
        xy, _ = base_pose(d, ix)
        c0 = time.perf_counter()
        home, info = plan_path(grid, xy, world.start[:2], np.pi)
        summary["plan_home"] = dict(plan_ms=1000 * (time.perf_counter() - c0), length_m=info["length"],
                                    min_clearance_m=info["min_clearance"])
        paths.append(("home", home, info))
        if video:
            video.markers = [(np.array([*p, 0.02]), [0.6, 0.3, 0.9, 1]) for p in home[::4]]
        summary["walk_home"] = walk(plant, d, ix, pol, PathFollower(home, np.pi), grid, video, log, "home")

    walking = [r for r in log]
    moving = [r for r in walking if np.linalg.norm(r["cmd"][:2]) > 0.2]
    summary["walking"] = dict(
        min_clearance_m=float(min(r["clearance"] for r in walking)),
        world_contacts=sorted({h for r in walking for h in r["hits"]}),
        base_height_m=[round(float(min(r["z"] for r in walking)), 3), round(float(np.mean([r["z"] for r in walking])), 3)],
        tilt_max_deg=float(max(r["tilt"] for r in walking)),
        speed_tracking_rms=float(np.sqrt(np.mean([np.sum((r["cmd"][:2] - r["v"][:2]) ** 2) for r in moving]))) if moving else None,
        total_time_s=float(d.time))
    print(json.dumps(summary, indent=1, default=float))
    name = f"{args.robot}{'_panel' if args.station == 'panel' else ''}{args.tag}"
    with open(os.path.join(out, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    plot_map(world, grid, paths, log, os.path.join(out, f"map_{name}.png"), est)
    if video and video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission_{name}.mp4"), video.frames, fps=FPS, quality=7, macro_block_size=1)


if __name__ == "__main__":
    main()
