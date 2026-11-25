"""ArUco work-order mission on the user's terrain (Go2 + Z1, MuJoCo).

The robot knows the map (walls, staircase, obstacles; no SLAM) and a list of work orders keyed by ArUco id. It does
NOT know where the markers are, where the pipes are or how big they are, or exactly where a wall face is.

    1. look around: the head RGB camera (OpenCV ArUco + PnP) finds the markers in sight
    2. queue: every confirmed marker becomes task(s); the order minimises A* travel cost (stairs = portal edge),
       task -> task directly, re-ordered whenever a new marker is confirmed (also while walking)
    3. nothing queued: patrol the map (waypoints down the corridor, over the stairs) while looking for markers
    4. at a task:
         paint (wall band): stand back, depth-camera scan -> wall plane; the band is split into patches the arm
             can trace from one stance; dock at each patch (sidestep along the wall), trace it, next patch
         pipe: stand in front of the sign, depth-camera scan -> cylinder fit (axis, radius) near the sign;
             walk to a pre-dock pose facing the pipe, re-scan up close, dock, trace +-35 deg x a height band
       Tracing: the stance WBC only stabilises the robot (base + feet), the arm runs a plain admittance loop on
       the measured contact force (5 N) + IK (sim/area_task.py). The trail on the real surface is shaded by the
       contact force: lighter = less force, darker = more (sim/paint.py).

Locomotion: blind MoE-CTS Go2 policy (sim/locomotion/blind_policy.py) - climbs the stairs with the arm on board.

    python sim/aruco_mission_demo.py                      # -> results/aruco_mission/
    python sim/aruco_mission_demo.py --no-video
    python sim/aruco_mission_demo.py --terrain-decimate 2 # 4 cm terrain cells (lighter rendering)
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from area_task import AreaTask, PipeArea, WallArea  # noqa: E402
from aruco_world import MARKER_SIZE, PATCH_MAX_WIDTH, WORK_ORDERS, aruco_world  # noqa: E402
from force_demo import ForceSensor  # noqa: E402
from locomotion.blind_policy import BlindPolicy  # noqa: E402
from paint import CylinderCanvas, PlaneCanvas  # noqa: E402
from perception.aruco_detect import ArucoCamera, MarkerMap, add_head_camera  # noqa: E402
from perception.depth_camera import DepthCamera  # noqa: E402
from perception.work_surfaces import fit_pipe_near, fit_wall_plane  # noqa: E402
from planning.follower import PathFollower, wrap  # noqa: E402
from planning.grid_planner import resample  # noqa: E402
from planning.task_planner import PortalPlanner  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402

DT = 0.002
DETECT_EVERY, FRAME_EVERY = 0.4, 0.08       # [s] sim time (video plays at 25 fps -> 2x speed)
D_WORK, D_SCAN_WALL, D_SCAN_PIPE = 0.85, 1.30, 1.35
PATROL = [(3.5, 0.0), (7.2, 0.0), (13.5, -0.6), (17.5, 0.0), (21.0, 0.0)]
F_DES = 5.0


@dataclass
class Task:
    key: str
    mid: int
    kind: str                  # "paint" | "pipe"
    pose: np.ndarray           # where to go first (x, y, yaw): wall scan pose / pipe viewpoint / next patch dock
    group: int
    patch: int = 0
    info: dict = field(default_factory=dict)


def base_pose(d, ix):
    R = d.xmat[ix.base].reshape(3, 3)
    return d.qpos[0:2].copy(), float(np.arctan2(R[1, 0], R[0, 0]))


class Mission:
    def __init__(self, args):
        self.args = args
        self.world = aruco_world(decimate=args.terrain_decimate)
        self.cfg = ROBOTS["go2_z1"]()
        self.m = build_model(self.cfg, world=self.world, contact_tip=True, scene_fn=add_head_camera,
                             render_size=(1280, 720))
        self.m.opt.timestep = DT
        self.ctrl = controller_model(self.cfg)
        self.d = mujoco.MjData(self.m)
        self.ix = Indexer(self.m, self.cfg)
        x, y, yaw = self.world.start
        initial_state(self.m, self.d, self.ix, xy=(x, y), yaw=yaw)
        self.d.qpos[2] += self.world.ground_z(x, y) + 0.01
        mujoco.mj_forward(self.m, self.d)
        self.pol = BlindPolicy(self.m, self.cfg, args.policy)
        self.planner = PortalPlanner(self.world)
        self.rgb = ArucoCamera(self.m, MARKER_SIZE)
        self.depth = DepthCamera(self.m)
        self.markers = MarkerMap(WORK_ORDERS.keys())
        self.sensor = ForceSensor(self.m)
        self.pending, self.done, self.failed = {}, [], []
        self.patrol_left = [np.array(p) for p in PATROL]
        self.events, self.walk_log, self.path_now, self.cloud = [], [], None, None
        self.t_detect, self.t_frame = -1.0, -1.0
        self.canvases = self._make_canvases()
        self.active_canvas = None
        self.summary = dict(tasks={}, markers={}, orders=[])
        self.video = None if args.no_video else Video(self.m, self.ix)
        self.mode_text = ""

    # ------------------------------------------------------------------ ground-truth paint canvases
    def _make_canvases(self):
        cv = {}
        for t in self.world.tasks:
            mk, wo = t.marker, WORK_ORDERS[t.marker.id]
            R = mk.frame()
            if wo["kind"] == "paint":
                c = np.asarray(mk.pos) + wo["area_offset"][0] * R[:, 0] + wo["area_offset"][1] * R[:, 1]
                W, H = wo["area_size"]
                mg = 0.08
                origin = c - (W / 2 + mg) * R[:, 0] - (H / 2 + mg) * R[:, 1]
                cv[mk.id] = PlaneCanvas(origin, R[:, 0], R[:, 1], R[:, 2], (W + 2 * mg, H + 2 * mg),
                                        (mg, mg, mg + W, mg + H), cell=0.006, color=wo["color"])
            else:
                p = t.pipe
                g = self.world.ground_z(*p.xy)
                cv[mk.id] = CylinderCanvas(p.xy, p.radius, (-1, 0), (-np.pi, np.pi), (g + 0.25, g + 0.75),
                                           (0, 0, 0, 0), cell=0.006, color=wo["color"])
        return cv

    def log(self, msg):
        self.events.append(dict(t=round(self.d.time, 2), msg=msg))
        print(f"[{self.d.time:6.1f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ perception: markers
    def detect(self):
        obs = self.rgb.detect(self.m, self.d)
        new = self.markers.update(obs)
        for mid in new:
            self.log(f"marker {mid} confirmed ({WORK_ORDERS[mid]['kind']}) at {np.round(self.markers.pose(mid)[0], 2)}")
            self._add_tasks(mid)
        if self.video is not None:
            self.video.inset(self.rgb, obs)
        return bool(new)

    def _add_tasks(self, mid):
        wo = WORK_ORDERS[mid]
        p, R = self.markers.pose(mid)
        n = np.array([*R[:2, 2], 0.0]); n /= np.linalg.norm(n)
        if wo["kind"] == "paint":
            c = p + wo["area_offset"][0] * R[:, 0] + wo["area_offset"][1] * R[:, 1]
            xy = self.free_near(c[:2] + D_SCAN_WALL * n[:2])
            self.pending[f"{mid}"] = Task(f"{mid}", mid, "paint", np.array([*xy, np.arctan2(-n[1], -n[0])]), mid)
        else:
            # viewpoint in front of the sign, pulled towards the corridor centre line, facing the sign
            vp = p[:2] + 1.6 * n[:2]
            vp[1] += np.clip(-vp[1], -0.5, 0.5)
            vp = self.free_near(vp)
            look = p[:2] - vp
            self.pending[f"{mid}"] = Task(f"{mid}", mid, "pipe", np.array([*vp, np.arctan2(look[1], look[0])]), mid)
            self.planner.add_obstacle(("circle", p[0], p[1], 0.12))         # the sign post
        self.reorder()

    def free_near(self, xy):
        g = self.planner.grid
        if g.free[g.to_cell(xy)]:
            return np.asarray(xy, float)
        from planning.grid_planner import _nearest_free
        return g.to_xy(_nearest_free(g, g.to_cell(xy)))

    def reorder(self):
        if not self.pending:
            return []
        xy, _ = base_pose(self.d, self.ix)
        c0 = time.perf_counter()
        keys, cost = self.planner.order(xy, [(k, t.pose, t.group) for k, t in self.pending.items()])
        self.queue = keys
        self.summary["orders"].append(dict(t=round(self.d.time, 2), queue=keys, cost_m=round(float(cost), 2),
                                           ms=round(1000 * (time.perf_counter() - c0), 1)))
        self.log(f"queue -> {keys} (travel cost {cost:.1f} m)")
        return keys

    # ------------------------------------------------------------------ walking
    def step_rl(self, cmd):
        self.pol.torques(self.m, self.d, cmd)
        mujoco.mj_step(self.m, self.d)
        R = self.d.xmat[self.ix.base].reshape(3, 3)
        if R[2, 2] < 0.5:
            raise RuntimeError(f"robot fell at t={self.d.time:.2f}s, x={self.d.qpos[0]:.2f}")
        return R

    def walk(self, path, goal_yaw, label, tol=(0.05, 4.0), interruptible=True, t_max=90.0, v_max=0.5):
        """Follow `path` with the RL policy. Returns 'arrived' | 'interrupted' (a new marker changed the queue head)."""
        self.path_now = path
        st = self.planner
        fol = PathFollower(path, goal_yaw, v_max=v_max, vy_max=0.3, wz_max=0.8, dock_radius=0.8)
        t0, t_near, cmd = self.d.time, None, np.zeros(3)
        head = self.queue[0] if getattr(self, "queue", None) else None
        while self.d.time - t0 < t_max:
            xy, yaw = base_pose(self.d, self.ix)
            if self.pol.k % 10 == 0:
                cmd = fol.command(xy, yaw)
                if st.on_stairs(xy):            # climb straight and slower
                    cmd[0] = min(cmd[0], 0.35); cmd[1] = 0.0
                near = np.linalg.norm(fol.goal - xy) < tol[0] and abs(wrap(goal_yaw - yaw)) < np.radians(tol[1])
                t_near = (t_near or self.d.time) if near else None
                if fol.done or (t_near is not None and self.d.time - t_near > 0.6):
                    break
            R = self.step_rl(cmd)
            self.walk_log.append((self.d.time, xy[0], xy[1], float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), label))
            if self.d.time - self.t_detect >= DETECT_EVERY:
                self.t_detect = self.d.time
                if self.detect() and interruptible and not st.on_stairs(xy):
                    if self.queue and self.queue[0] != head:
                        self.log(f"queue head changed {head} -> {self.queue[0]}: re-planning")
                        return "interrupted"
            self.frame(f"{label}", cmd)
        self.stand(0.6)
        return "arrived"

    def stand(self, T, on_tick=None):
        t0 = self.d.time
        while self.d.time - t0 < T:
            self.step_rl(np.zeros(3))
            if on_tick is not None and self.pol.k % 10 == 0:
                on_tick()
            self.frame("standing", np.zeros(3))

    def goto(self, pose, label, approach=0.8, **kw):
        xy, _ = base_pose(self.d, self.ix)
        c0 = time.perf_counter()
        path, info = self.planner.plan(xy, pose[:2], pose[2], approach=approach)
        self.log(f"A* to {label}: {info['length']:.1f} m{' via stairs' if info['stairs'] else ''} "
                 f"({1000 * (time.perf_counter() - c0):.0f} ms)")
        return self.walk(path, pose[2], label, **kw)

    def line_to(self, pose, label, tol=(0.04, 3.0)):
        xy, _ = base_pose(self.d, self.ix)
        return self.walk(resample(np.array([xy, pose[:2]]), 0.05), pose[2], label, tol=tol, interruptible=False, t_max=20)

    def look_around(self):
        self.mode_text = "look around"
        for wz, T in [(0.6, 1.8), (-0.6, 3.6), (0.6, 1.8)]:
            t0 = self.d.time
            while self.d.time - t0 < T:
                self.step_rl(np.array([0.0, 0.0, wz]))
                if self.d.time - self.t_detect >= DETECT_EVERY / 2:
                    self.t_detect = self.d.time
                    self.detect()
                self.frame("look around", np.array([0, 0, wz]))
        self.stand(0.5)

    # ------------------------------------------------------------------ depth scans
    def depth_scan(self, frames=8):
        clouds = []
        self.stand(0.4)
        self.stand(1.2, on_tick=lambda: clouds.append(self.depth.capture(self.m, self.d)[:2]) if len(clouds) < frames else None)
        return np.concatenate([c[0] for c in clouds]), clouds[-1][1]

    # ------------------------------------------------------------------ tasks
    def run_task(self, task):
        if task.kind == "paint":
            return self.task_paint(task)
        return self.task_pipe(task)

    def task_paint(self, task):
        mid, wo = task.mid, WORK_ORDERS[task.mid]
        if "plane" not in task.info:              # first visit: scan the wall
            if self.goto(task.pose, f"wall {mid}: scan pose", approach=0.6) == "interrupted":
                return "interrupted"
            p, R = self.markers.pose(mid)
            c = p + wo["area_offset"][0] * R[:, 0] + wo["area_offset"][1] * R[:, 1]
            pts, cam = self.depth_scan()
            pl = fit_wall_plane(pts, cam, c)
            self.cloud = (pts[::4], [0.2, 1.0, 0.3, 1])
            n = pl.normal
            c_on = c - ((c - pl.point) @ n) * n          # area centre projected onto the measured plane
            u = np.cross(-n, [0, 0, 1.0]); u /= np.linalg.norm(u)
            W, H = wo["area_size"]
            k = int(np.ceil(W / PATCH_MAX_WIDTH - 1e-9))
            w = W / k
            true_mk = next(t.marker for t in self.world.tasks if t.marker.id == mid)
            n_true = np.asarray(true_mk.normal, float)
            d_true = (c_on - np.asarray(true_mk.pos)) @ n_true
            self.log(f"wall {mid}: plane fit {pl.inliers} pts, rms {pl.rms_mm:.1f} mm, normal err "
                     f"{np.degrees(np.arccos(np.clip(n @ n_true, -1, 1))):.2f} deg, face err {1000 * d_true:.1f} mm "
                     f"-> {k} patches of {w:.2f} m")
            task.info.update(plane=pl, c=c_on, u=u, n=n, w=w, H=H, k=k, perception=dict(
                inliers=pl.inliers, rms_mm=pl.rms_mm, normal_err_deg=float(np.degrees(np.arccos(np.clip(n @ n_true, -1, 1)))),
                face_err_mm=float(1000 * d_true)))
            # queue the patches as consecutive tasks of this group (sidestep along the wall)
            self.pending.pop(task.key, None)
            for i in range(k):
                pc = c_on + ((i + 0.5) * w - W / 2) * u
                area_i = WallArea(pc - w / 2 * u - H / 2 * np.array([0, 0, 1.0]), n, u, w, H)
                try:
                    dock = self.choose_standoff(lambda: area_i, lambda D: np.array([*(pc[:2] + D * n[:2]), np.arctan2(-n[1], -n[0])]),
                                                f"wall {mid} patch {i + 1}")
                except RuntimeError as ex:
                    self.log(str(ex))
                    continue
                self.pending[f"{mid}.{i}"] = Task(f"{mid}.{i}", mid, "paint", dock, mid, patch=i, info=task.info)
            self.reorder()
            return "queued"
        # a patch: dock (line if we are next to it already) and trace
        xy, _ = base_pose(self.d, self.ix)
        if np.linalg.norm(xy - task.pose[:2]) < 1.2:
            self.line_to(task.pose, f"wall {mid}: patch {task.patch + 1}/{task.info['k']} sidestep")
        elif self.goto(task.pose, f"wall {mid}: patch {task.patch + 1}", approach=0.6) == "interrupted":
            return "interrupted"
        else:
            self.line_to(task.pose, f"wall {mid}: patch dock")
        I = task.info
        pc = I["c"] + ((task.patch + 0.5) * I["w"] - (I["k"] * I["w"]) / 2) * I["u"]
        area = WallArea(pc - I["w"] / 2 * I["u"] - I["H"] / 2 * np.array([0, 0, 1.0]), I["n"], I["u"], I["w"], I["H"])
        return self.trace(task, area)

    def task_pipe(self, task):
        mid, wo = task.mid, WORK_ORDERS[task.mid]
        if self.goto(task.pose, f"pipe {mid}: viewpoint", approach=0.6) == "interrupted":
            return "interrupted"
        p, R = self.markers.pose(mid)
        true = next(t for t in self.world.tasks if t.marker.id == mid).pipe
        g = self.world.ground_z(*p[:2])
        zw = g + np.mean(wo["band"])
        pts, cam = self.depth_scan()
        est = fit_pipe_near(pts, cam, p, R[:, 2], wo["search_radius"], g, zw)
        self.cloud = (est.points, [0.2, 1.0, 0.3, 1])
        self.log(f"pipe {mid}: coarse fit r={1000 * est.radius:.0f} mm, axis err "
                 f"{1000 * np.linalg.norm(est.xy - true.xy):.0f} mm (real r={1000 * true.radius:.0f})")
        self.planner.add_obstacle(("circle", est.xy[0], est.xy[1], est.radius + 0.05))
        xy, _ = base_pose(self.d, self.ix)
        e = (xy - est.xy) / np.linalg.norm(xy - est.xy)
        yaw = np.arctan2(-e[1], -e[0])
        pre = np.array([*(est.xy + D_SCAN_PIPE * e), yaw])
        self.goto(pre, f"pipe {mid}: pre-dock", approach=0.5, interruptible=False)
        pts, cam = self.depth_scan()
        est2 = fit_pipe_near(pts, cam, np.array([*est.xy, zw]), np.array([*e, 0]), 0.5, g, zw, remove_sign=False)
        self.cloud = (est2.points, [0.2, 1.0, 0.3, 1])
        err = 1000 * np.linalg.norm(est2.xy - true.xy)
        self.log(f"pipe {mid}: close fit r={1000 * est2.radius:.1f} mm (real {1000 * true.radius:.0f}), axis err "
                 f"{err:.1f} mm, tilt {est2.tilt_deg:.2f} deg, rms {est2.rms_mm:.1f} mm")
        task.info["perception"] = dict(coarse_axis_err_mm=float(1000 * np.linalg.norm(est.xy - true.xy)),
                                       coarse_radius_mm=float(1000 * est.radius), axis_err_mm=float(err),
                                       radius_mm=float(1000 * est2.radius), radius_true_mm=float(1000 * true.radius),
                                       tilt_deg=est2.tilt_deg, rms_mm=est2.rms_mm, inliers=est2.inliers)
        xy, _ = base_pose(self.d, self.ix)
        e = (xy - est2.xy) / np.linalg.norm(xy - est2.xy)
        th = np.radians(wo["arc_deg"])
        area = PipeArea(est2.xy, est2.radius, e, th, g + wo["band"][0], g + wo["band"][1])
        try:
            dock = self.choose_standoff(lambda: area, lambda D: np.array([*(est2.xy + D * e), np.arctan2(-e[1], -e[0])]),
                                        f"pipe {mid}")
        except RuntimeError as ex:
            self.log(str(ex))
            return "failed"
        task.pose = dock
        self.line_to(dock, f"pipe {mid}: dock", tol=(0.03, 2.5))
        # score on the real pipe: the band the robot planned, mapped onto the real surface
        cv = self.canvases[mid]
        ref_true = np.arctan2(*(cv.e2[:2] @ e, cv.e1[:2] @ e))
        TH = cv.th0 + (np.arange(cv.r.shape[0]) + 0.5) * cv.dth
        Z = cv.z0 + (np.arange(cv.r.shape[1]) + 0.5) * cv.cell
        dth = np.abs(wrap(TH - ref_true))
        cv.target = (dth[:, None] <= th) & (Z[None, :] >= area.z_lo) & (Z[None, :] <= area.z_lo + area.height)
        return self.trace(task, area)

    def virtual_qpos(self, pose):
        """Nominal stance (arm stowed) at a base pose on the local ground - for reachability checks before docking."""
        dv = mujoco.MjData(self.ctrl)
        initial_state(self.ctrl, dv, Indexer(self.ctrl, self.cfg), xy=pose[:2], yaw=pose[2])
        dv.qpos[2] += self.world.ground_z(*pose[:2])
        return dv.qpos.copy()

    def choose_standoff(self, make_area, make_pose, label):
        """Try standoffs (closest to D_WORK first) until the whole raster is reachable from that stance."""
        for D in (D_WORK, 0.80, 0.90, 0.75, 0.95):
            pose = make_pose(D)
            try:
                at = AreaTask(self.cfg, self.ctrl, self.virtual_qpos(pose), make_area(), t0=0.0, f_des=F_DES)
                self.log(f"{label}: standoff {D:.2f} m reachable (IK margin {at.ik_margin:.2f})")
                return pose
            except RuntimeError:
                continue
        raise RuntimeError(f"{label}: no reachable standoff")

    def settle(self, t_max=4.0):
        """Stand (RL, zero command) until all four feet are planted and the base is still for 0.3 s: the stance WBC
        assumes four fixed feet, so it must not take over mid-step."""
        feet = set(self.ix.feet.tolist())
        t0, t_ok = self.d.time, None
        while self.d.time - t0 < t_max:
            self.step_rl(np.zeros(3))
            self.frame("settling", np.zeros(3))
            touching = {c.geom1 for c in self.d.contact[:self.d.ncon]} | {c.geom2 for c in self.d.contact[:self.d.ncon]}
            still = np.linalg.norm(self.d.qvel[0:3]) < 0.04 and np.linalg.norm(self.d.qvel[3:6]) < 0.25
            ok = feet <= touching and still
            t_ok = (t_ok or self.d.time) if ok else None
            if t_ok is not None and self.d.time - t_ok > 0.3:
                return True
        return False

    def trace(self, task, area):
        """Hand-off RL -> stance WBC (base) + arm admittance; trace the area; hand back to RL."""
        mid = task.mid
        cv = self.canvases[mid]
        self.active_canvas = cv
        at = None
        for attempt in range(2):
            if not self.settle():
                self.log(f"task {task.key}: did not settle on four feet")
            xy, yaw = base_pose(self.d, self.ix)
            self.log(f"task {task.key}: stance at {np.round(xy, 3)}, dock error "
                     f"{1000 * np.linalg.norm(xy - task.pose[:2]):.0f} mm / {np.degrees(abs(wrap(yaw - task.pose[2]))):.1f} deg")
            try:
                at = AreaTask(self.cfg, self.ctrl, self.d.qpos.copy(), area, t0=self.d.time, f_des=F_DES)
                break
            except RuntimeError as ex:
                self.log(f"task {task.key}: {ex}" + (" -> re-dock" if attempt == 0 else ""))
                if attempt == 0:
                    self.line_to(task.pose, f"re-dock {task.key}", tol=(0.02, 2.0))
        if at is None:
            return "failed"
        wbc = StanceWBC(self.ctrl, self.cfg, **self.cfg.wbc_gains)
        self.sensor = ForceSensor(self.m)
        base0 = self.d.qpos[0:3].copy()
        xy, yaw = base_pose(self.d, self.ix)
        dock_err = (1000 * float(np.linalg.norm(xy - task.pose[:2])), float(np.degrees(abs(wrap(yaw - task.pose[2])))))
        self.log(f"hand-off at {np.round(xy, 3)}: dock error {dock_err[0]:.0f} mm / {dock_err[1]:.1f} deg, "
                 f"base speed {np.linalg.norm(self.d.qvel[0:3]):.2f} m/s")
        F, drift = [], 0.0
        self.log(f"task {task.key}: stance WBC + arm admittance, {len(at.segs)} raster segments, "
                 f"IK margin {at.ik_margin:.2f}, {at.T:.0f} s")
        while self.d.time < at.t0 + at.T:
            f = self.sensor.read(self.m, self.d)
            ref = at.ref(self.d.time, self.d.qpos.copy(), f)
            tau, _ = wbc.compute(self.d.qpos.copy(), self.d.qvel.copy(), ref)
            self.d.ctrl[:] = tau[self.ix.act_dof]
            mujoco.mj_step(self.m, self.d)
            ph = at.phase(self.d.time)
            cv.stamp(self.d.site_xpos[self.ix.tip].copy(), self.sensor.raw, F_DES)
            if self.d.xmat[self.ix.base][8] < 0.85:
                raise RuntimeError(f"robot tipped over during task {task.key} (phase {ph}, t={self.d.time:.2f}s)")
            if ph == "raster":
                F.append(self.sensor.raw)
                drift = max(drift, float(np.linalg.norm(self.d.qpos[0:2] - base0[0:2])))
            self.frame(f"work {task.key}: {ph}", None, work=area, force=f, prog=at.progress(self.d.time))
        self.pol.reset()
        self.active_canvas = None
        F = np.array(F)
        res = dict(kind=task.kind, patch=task.patch, force_mean=float(F.mean()), force_rms_err=float(np.sqrt(np.mean((F - F_DES) ** 2))),
                   force_min_max=[float(F.min()), float(F.max())], contact_lost=float(np.mean(F < 0.5)),
                   base_drift_mm=1000 * drift, wbc_qp_failures=int(wbc.fail), ik_margin=at.ik_margin,
                   approach_wait_s=round(at.wait_s, 2), perception=task.info.get("perception"))
        self.summary["tasks"][task.key] = res
        res["dock_err_mm_deg"] = dock_err
        ok = res["contact_lost"] < 0.5
        self.log(f"task {task.key} {'done' if ok else 'FAILED (no contact)'}: F {res['force_mean']:.2f} N "
                 f"(rms err {res['force_rms_err']:.2f}), contact lost {100 * res['contact_lost']:.1f} %")
        return "done" if ok else "failed"

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.log(f"start: work orders {sorted(WORK_ORDERS)}, map known, marker positions unknown")
        self.queue = []
        self.look_around()
        while True:
            if self.pending:
                if not self.queue or self.queue[0] not in self.pending:
                    self.reorder()
                task = self.pending[self.queue[0]]
                r = self.run_task(task)
                if r in ("done", "failed"):
                    self.pending.pop(task.key, None)
                    (self.done if r == "done" else self.failed).append(task.key)
                    self.reorder()
                elif r == "interrupted":
                    continue
                continue
            xy, _ = base_pose(self.d, self.ix)
            self.patrol_left = [w for w in self.patrol_left if np.linalg.norm(w - xy) > 1.2 and w[0] > xy[0] - 1.0]
            if not self.patrol_left:
                break
            w = self.patrol_left[0]
            self.mode_text = "patrol"
            self.goto(np.array([*w, 0.0]), "patrol", approach=0.0, tol=(0.25, 25.0))
            self.patrol_left = self.patrol_left[1:] if np.linalg.norm(base_pose(self.d, self.ix)[0] - w) < 0.6 else self.patrol_left
        self.log(f"mission complete: done {self.done}, failed {self.failed}")
        self.stand(1.5)

    # ------------------------------------------------------------------ video
    def frame(self, label, cmd, work=None, force=None, prog=None):
        if self.video is None or self.d.time - self.t_frame < FRAME_EVERY - 1e-9:
            return
        self.t_frame = self.d.time
        q = " > ".join(self.queue[:6]) if getattr(self, "queue", None) else "-"
        l1 = f"t={self.d.time:6.1f}s  {label}"
        if force is not None:
            l1 += f"   F = {force:4.1f} N (target {F_DES:.0f})  {100 * prog:3.0f} %"
        l2 = f"queue: {q}   done: {', '.join(self.done) or '-'}   markers: {self.markers.confirmed}"
        self.video.frame(self.m, self.d, [l1, l2], self.canvases.values(), self.path_now, work, self.cloud if work is None else None)


class Video:
    def __init__(self, m, ix):
        import cv2
        self.cv2 = cv2
        self.r = mujoco.Renderer(m, 540, 960, max_geom=30000)
        self.track = mujoco.MjvCamera()
        self.track.type, self.track.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, ix.base
        self.track.distance, self.track.elevation = 3.2, -28
        self.work = mujoco.MjvCamera()
        self.ix, self.frames, self.inset_img = ix, [], None

    def inset(self, rgb, obs):
        cv2 = self.cv2
        img = cv2.resize(rgb.last_image, (320, 180))
        s = 320 / rgb.last_image.shape[1]
        corners, ids, _ = rgb.det.detectMarkers(cv2.cvtColor(rgb.last_image, cv2.COLOR_RGB2GRAY))
        if ids is not None:
            img = np.ascontiguousarray(img)
            cv2.aruco.drawDetectedMarkers(img, [c * s for c in corners], ids, (0, 255, 0))
        cv2.rectangle(img, (0, 0), (319, 179), (255, 255, 255), 1)
        cv2.putText(img, "head RGB: ArUco", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.inset_img = img

    def frame(self, m, d, lines, canvases, path, work, cloud):
        from PIL import Image, ImageDraw
        R = d.xmat[self.ix.base].reshape(3, 3)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        if work is not None:
            c, _ = work.point(work.width / 2, work.height / 2)
            self.work.lookat[:] = c
            self.work.distance, self.work.azimuth, self.work.elevation = 1.5, yaw + 38, -18
            cam = self.work
        else:
            self.track.azimuth = yaw + 25
            cam = self.track
        self.r.update_scene(d, camera=cam)
        scn = self.r.scene
        for cv in canvases:
            cv.draw(scn)
        if path is not None and work is None:
            for p in path[::5]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
                                    np.array([p[0], p[1], 0.11]), np.eye(3).ravel(), np.array([0.1, 0.5, 1.0, 0.8], np.float32))
                scn.ngeom += 1
        if cloud is not None:
            pts, rgba = cloud
            for p in pts[:: max(1, len(pts) // 1500)]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], p,
                                    np.eye(3).ravel(), np.array(rgba, np.float32))
                scn.ngeom += 1
        im = self.r.render().copy()
        if self.inset_img is not None:
            im[8:188, -328:-8] = self.inset_img
        pil = Image.fromarray(im)
        dr = ImageDraw.Draw(pil)
        dr.rectangle([0, im.shape[0] - 36, im.shape[1], im.shape[0]], fill=(0, 0, 0))
        dr.multiline_text((6, im.shape[0] - 33), "\n".join(lines), fill=(255, 255, 255))
        self.frames.append(np.asarray(pil))


def plot_map(ms, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, g = ms.world, ms.planner.grid
    fig, ax = plt.subplots(figsize=(15, 4.2))
    x0, x1, y0, y1 = w.bounds
    ax.imshow(np.where(g.occ, 2, np.where(g.free, 0, 1)).T, origin="lower", extent=[x0, x1, y0, y1], cmap="Greys", vmax=2.6)
    st = w.stairs
    ax.add_patch(plt.Rectangle((st.x_start, st.y_lo), st.x_end - st.x_start, st.y_hi - st.y_lo, color="#bf8700", alpha=0.25))
    ax.annotate("stairs (portal)", ((st.x_start + st.x_end) / 2, 0), ha="center", fontsize=8)
    a, b = ms.planner.portal
    ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#bf8700")
    tr = np.array([(r[1], r[2]) for r in ms.walk_log])
    if len(tr):
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#1f6feb", lw=1.2, label="walked")
    for t in w.tasks:
        mk = t.marker
        ax.plot(mk.pos[0], mk.pos[1], "ks", ms=6)
        ax.annotate(f" {mk.id}", mk.pos[:2], fontsize=8)
        if mk.id in ms.markers.est:
            p, _ = ms.markers.pose(mk.id)
            ax.plot(p[0], p[1], "x", color="#cf222e", ms=8)
        if t.pipe is not None:
            ax.add_patch(plt.Circle(t.pipe.xy, t.pipe.radius, color="k", fill=False))
    ax.plot([], [], "ks", label="marker (true)"); ax.plot([], [], "x", color="#cf222e", label="marker (ArUco + PnP)")
    ax.plot(*w.start[:2], "go", ms=8, label="start")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Known map (grey: obstacles + robot-radius margin) - walked path - markers found")
    fig.tight_layout(); fig.savefig(path_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="moe_cts", choices=["moe_cts", "symmetry"])
    ap.add_argument("--terrain-decimate", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "aruco_mission")
    os.makedirs(out, exist_ok=True)
    ms = Mission(args)
    c0 = time.time()
    err = None
    try:
        ms.run()
    except RuntimeError as ex:
        err = str(ex)
        ms.log(f"ABORTED: {ex}")
    S = ms.summary
    tilt = np.array([r[3] for r in ms.walk_log]) if ms.walk_log else np.zeros(1)
    for mid in ms.markers.confirmed:
        p, R = ms.markers.pose(mid)
        mk = next(t.marker for t in ms.world.tasks if t.marker.id == mid)
        Rg = mk.frame()
        S["markers"][str(mid)] = dict(sightings=ms.markers.est[mid].n,
                                     pos_err_mm=float(1000 * np.linalg.norm(p - (np.asarray(mk.pos) + 0.006 * Rg[:, 2]))),
                                     normal_err_deg=float(np.degrees(np.arccos(np.clip(R[:, 2] @ Rg[:, 2], -1, 1)))))
    S["paint"] = {str(k): v.metrics() for k, v in ms.canvases.items()}
    S.update(done=ms.done, failed=ms.failed, aborted=err, sim_time_s=float(ms.d.time), wall_time_s=time.time() - c0,
             walking=dict(max_tilt_deg=float(tilt.max()), distance_m=float(np.sum(np.linalg.norm(np.diff(
                 np.array([(r[1], r[2]) for r in ms.walk_log]), axis=0), axis=1))) if len(ms.walk_log) > 1 else 0.0),
             events=ms.events, policy=args.policy, terrain_decimate=args.terrain_decimate)
    json.dump(S, open(os.path.join(out, f"summary{args.tag}.json"), "w"), indent=2, default=float)
    plot_map(ms, os.path.join(out, f"map{args.tag}.png"))
    if ms.video is not None and ms.video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission{args.tag}.mp4"), ms.video.frames, fps=25, quality=7, macro_block_size=1)
    print(json.dumps({k: S[k] for k in ("done", "failed", "aborted", "markers", "paint", "tasks")}, indent=1, default=float))


if __name__ == "__main__":
    main()
