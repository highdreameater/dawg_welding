"""Full mission in the workshop world: plan -> walk (RL) -> dock -> force-controlled pass on the pipe (WBC) -> walk home.

    start (0, 0) --A* path--> pipe station --RL policy + path follower--> dock (head-on, ~cm accuracy)
        -> hand-off: stance WBC braces, unstows, presses 5 N on the pipe while sweeping +-30 deg, stows
        -> hand-back: RL policy, planned path back to the start

    python sim/mission_demo.py                     # -> results/mission/ (video, map, summary)
    python sim/mission_demo.py --force 8 --no-return
    python sim/mission_demo.py --no-video          # ~3x faster

Locomotion is the pretrained Isaac Lab ANYmal-C rough-terrain policy run in MuJoCo (sim/locomotion/anymal_rl.py),
with the arm stowed as payload. The map is known (sim/world.py) - no SLAM.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from force_demo import ForceSensor, metrics as force_metrics  # noqa: E402
from force_task import PipeForceTask  # noqa: E402
from locomotion.anymal_rl import SIM_DT as RL_DT, AnymalRoughPolicy  # noqa: E402
from planning.follower import PathFollower, wrap  # noqa: E402
from planning.grid_planner import OccupancyGrid, plan_path  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import WORLDS  # noqa: E402

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
        self.track.distance, self.track.azimuth, self.track.elevation = 3.4, 225, -35
        self.work = mujoco.MjvCamera()
        self.work.lookat[:] = work_cam["lookat"]
        self.work.distance, self.work.azimuth, self.work.elevation = work_cam["distance"], work_cam["azimuth"], work_cam["elevation"]
        self.frames, self.markers, self.last = [], [], -1.0

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


def walk(m, d, ix, pol, follower, grid, video, log, label, t_max=60.0, stand=1.0):
    """Follow the path with the RL policy until docked (or timeout), then stand still for `stand` seconds."""
    m.opt.timestep = RL_DT
    pol.reset()
    foot_geoms = set(ix.feet.tolist())
    t0, t_near, t_done = d.time, None, None
    while d.time - t0 < t_max:
        xy, yaw = base_pose(d, ix)
        cmd = follower.command(xy, yaw) if t_done is None else np.zeros(3)
        if t_done is None:
            # the policy cannot servo to millimetres: accept a dock that stays within 6 cm / 4 deg for 1 s
            near = np.linalg.norm(follower.goal - xy) < 0.06 and abs(wrap(follower.goal_yaw - yaw)) < np.radians(4)
            t_near = (t_near or d.time) if near else None
            if follower.done or (t_near is not None and d.time - t_near > 1.0):
                follower.done, t_done = True, d.time
        elif d.time - t_done > stand:
            break
        pol.torques(m, d, cmd)
        mujoco.mj_step(m, d)
        hits = world_collisions(m, d, foot_geoms)
        log.append(dict(t=d.time, mode=label, x=xy[0], y=xy[1], yaw=yaw, cmd=cmd.copy(), z=d.qpos[2],
                        clearance=grid.clearance_at(xy), hits=sorted(hits)))
        if d.qpos[2] < 0.25:
            raise RuntimeError(f"robot fell while walking ({label}) at t={d.time:.2f}s")
        if video is not None and video.due(d.time):
            video.frame(d, f"MISSION  [{label}: {'walking' if t_done is None else 'standing'}]  t={d.time:5.1f}s\n"
                           f"cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}   "
                           f"to goal {np.linalg.norm(follower.goal - xy):.2f} m")
    xy, yaw = base_pose(d, ix)
    return dict(time_s=d.time - t0, docked=bool(follower.done),
                dock_pos_err_mm=1000 * float(np.linalg.norm(follower.goal - xy)),
                dock_yaw_err_deg=float(np.degrees(abs(wrap(follower.goal_yaw - yaw)))))


def force_phase(plant, ctrl, d, ix, cfg, st, args, video):
    plant.opt.timestep = WBC_DT
    task = PipeForceTask(cfg, ctrl, d.qpos.copy(), st.work, f_des=args.force, theta_max=np.radians(args.arc_deg),
                         passes=args.passes, speed=args.speed, t0=d.time)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    L = {k: [] for k in ["t", "f", "f_raw", "f_des", "phase", "tip", "tip_ref", "n", "axis", "base_pos", "base_quat",
                         "feet", "tau", "vn_des"]}
    solve_ms = []
    for _ in range(int(round(task.T / WBC_DT))):
        t = d.time
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas)
        c0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - c0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        ph = task.phase(t)
        n = ref["force"]["n"] if "force" in ref else st.work.surface_point(task.x_work, task.theta(t - task.t0))[1]
        L["t"].append(t - task.t0); L["f"].append(f_meas); L["f_raw"].append(sensor.raw); L["phase"].append(ph)
        L["f_des"].append(ref["force"]["f_des"] if "force" in ref else np.nan)
        L["tip"].append(d.site_xpos[ix.tip].copy()); L["tip_ref"].append(ref.get("ee_pos", np.full(3, np.nan)))
        L["n"].append(n); L["axis"].append(d.site_xmat[ix.tip].reshape(3, 3)[:, 0].copy())
        L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
        L["feet"].append(d.geom_xpos[ix.feet].copy()); L["tau"].append(d.actuator_force.copy())
        L["vn_des"].append(wbc.vn_des if "force" in ref else np.nan)
        if video is not None and video.due(d.time):
            video.frame(d, f"MISSION  [stance WBC + force control: {ph}]  t={d.time:5.1f}s\n"
                           f"F = {f_meas:5.2f} N (target {args.force:.1f})   arc "
                           f"{np.degrees(task.theta(t - task.t0)):+5.1f} deg", cam="work")
    L = {k: (np.array(x) if k != "phase" else np.array(x, dtype=object)) for k, x in L.items()}
    return force_metrics(L, task, wbc, plant, solve_ms, args)


def plot_map(world, grid, paths, log, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ext = list(world.bounds)
    ax.imshow(np.where(grid.occ, 2, np.where(grid.free, 0, 1)).T, origin="lower", extent=ext, cmap="Greys", vmax=2.6)
    for (label, p, info), col in zip(paths, ["#1f6feb", "#8250df"]):
        ax.plot(info["raw"][:, 0], info["raw"][:, 1], ":", color=col, lw=1)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2, alpha=0.5, label=f"planned ({label})")
    for label, col in [("to_pipe", "#cf222e"), ("home", "#bf8700")]:
        tr = np.array([(r["x"], r["y"]) for r in log if r["mode"] == label])
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.2, label=f"executed ({label})")
    for name, s in world.stations.items():
        ax.plot(*s.base_xy, "k^", ms=8)
        ax.annotate(f" {name} station", s.base_xy, fontsize=8)
    ax.plot(*world.start[:2], "ko", ms=7)
    ax.annotate(" start", world.start[:2], fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Workshop map: obstacles (black), robot-radius C-space margin (grey), A* raw (dotted) / smoothed paths")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="anymal_c_z1")
    ap.add_argument("--world", choices=list(WORLDS), default="workshop")
    ap.add_argument("--force", type=float, default=5.0)
    ap.add_argument("--arc-deg", type=float, default=30.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.03)
    ap.add_argument("--force-noise", type=float, default=0.0)
    ap.add_argument("--no-return", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg, world = ROBOTS[args.robot](), WORLDS[args.world]()
    args.torch_mass = args.ctrl_torch_mass = 1.2
    st = world.stations["pipe"]
    out = os.path.join(ROOT, "results", "mission")
    os.makedirs(out, exist_ok=True)

    plant = build_model(cfg, world=world, contact_tip=True)
    ctrl = controller_model(cfg)
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    initial_state(plant, d, ix, xy=world.start[:2], yaw=world.start[2])
    pol = AnymalRoughPolicy(plant, cfg)
    grid = OccupancyGrid(world)
    video = None if args.no_video else Video(
        plant, ix, dict(lookat=[st.work.x0 - 0.17, st.work.y, 0.45], distance=1.9, azimuth=200, elevation=-22))
    log, summary, paths = [], dict(robot=args.robot, world=args.world), []

    # 1) plan + walk to the pipe station
    c0 = time.perf_counter()
    path, info = plan_path(grid, world.start[:2], st.base_xy, st.base_yaw)
    summary["plan_to_pipe"] = dict(plan_ms=1000 * (time.perf_counter() - c0), length_m=info["length"],
                                   raw_length_m=info["raw_len"], expanded=info["expanded"],
                                   min_clearance_m=info["min_clearance"])
    paths.append(("to pipe", path, info))
    if video:
        video.markers = [(np.array([*p, 0.02]), [0.1, 0.5, 1.0, 1]) for p in path[::4]]
    summary["walk_to_pipe"] = walk(plant, d, ix, pol, PathFollower(path, st.base_yaw), grid, video, log, "to_pipe")

    # 2) hand-off to the stance WBC: force-controlled pass over the pipe
    if video:
        video.markers = []
    summary["force"] = force_phase(plant, ctrl, d, ix, cfg, st, args, video)

    # 3) hand back to the RL policy and walk home
    if not args.no_return:
        xy, _ = base_pose(d, ix)
        c0 = time.perf_counter()
        home, info = plan_path(grid, xy, world.start[:2], np.pi)
        summary["plan_home"] = dict(plan_ms=1000 * (time.perf_counter() - c0), length_m=info["length"],
                                    min_clearance_m=info["min_clearance"])
        paths.append(("home", home, info))
        if video:
            video.markers = [(np.array([*p, 0.02]), [0.6, 0.3, 0.9, 1]) for p in home[::4]]
        summary["walk_home"] = walk(plant, d, ix, pol, PathFollower(home, np.pi), grid, video, log, "home")

    walk_log = [r for r in log]
    summary["walking"] = dict(
        min_clearance_m=float(min(r["clearance"] for r in walk_log)),
        world_contacts=sorted({h for r in walk_log for h in r["hits"]}),
        min_base_height_m=float(min(r["z"] for r in walk_log)),
        total_time_s=float(d.time))
    print(json.dumps(summary, indent=1))
    with open(os.path.join(out, f"summary{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_map(world, grid, paths, log, os.path.join(out, f"map{args.tag}.png"))
    if video and video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission{args.tag}.mp4"), video.frames, fps=FPS, quality=7, macro_block_size=1)


if __name__ == "__main__":
    main()
