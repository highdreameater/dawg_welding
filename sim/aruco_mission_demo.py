"""ArUco work-order mission on the user's terrain (Go2 + Z1, MuJoCo) - one generic pipeline for every surface.

The robot knows the map (walls, staircase, obstacles; no SLAM) and a list of work orders keyed by ArUco id. A work
order is only "cover this rectangle next to the marker" (offset + size in the marker frame). It does not say what the
surface is. The robot does not know where the markers are or what the surfaces look like, and its pose estimate drifts.

    1. look around / patrol (waypoints sampled from the map's free space) - the head RGB camera (OpenCV ArUco + PnP)
       finds markers; the queue order minimises A* travel cost (stairs = portal edge), re-ordered on every new marker
    2. localisation: leg/IMU odometry that drifts (sim/perception/odometry.py); every marker is a landmark and later
       sightings pull the estimate back onto it - while walking, while docking and while standing at the surface
    3. survey: stand in front of the area, depth camera looking straight at it (all work surfaces are vertical),
       one non-parametric surface model w(u, v) over the area (sim/perception/work_area.py). Flat wall, pipe or wavy
       panel - same code, no shape fit
    4. reachability plans the work (sim/area_task.py): IK over the measured surface picks the standoff and splits the
       area into patches that the arm reaches from one stance (with margin for dock error)
    5. each patch: dock (re-localising on the marker), check reach from the actual stance (what is out of reach is
       queued as a new patch), then trace: the stance WBC only stabilises the quadruped, the arm runs admittance on the
       measured contact force (5 N) along the fitted normal + IK
    6. the trail on the real surface is an intensity brush: lighter = less force, darker = more (sim/paint.py)

Locomotion: blind MoE-CTS Go2 policy (sim/locomotion/blind_policy.py) - climbs the stairs with the arm on board.
Robot: go2_z1_official (Unitree mount, no torch: a contact tip 5 cm past the flange) - the same configuration the stair
climbing was validated on (sim/course_demo.py, sim/terrain_demo.py).

    python sim/aruco_mission_demo.py                      # -> results/aruco_mission/
    python sim/aruco_mission_demo.py --no-video
    python sim/aruco_mission_demo.py --panel-seed 2 --pipe-radius 0.10 0.12 --tag _b
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
from area_task import TRACE_WBC_GAINS, AreaTask, SurfacePatch, plan_stance, reachable_span, stance_pose  # noqa: E402
from aruco_world import MARKER_SIZE, SITE_HINTS, SITE_ORDER, WORK_ORDERS, aruco_world  # noqa: E402
from force_demo import ForceSensor  # noqa: E402
from locomotion.blind_policy import BlindPolicy  # noqa: E402
from paint import SurfaceCanvas  # noqa: E402
from perception.aruco_detect import ArucoCamera, MarkerEstimate, MarkerMap, add_head_camera  # noqa: E402
from perception.depth_camera import CameraMount, DepthCamera  # noqa: E402
from perception.odometry import Odometry, wrap  # noqa: E402
from perception.work_area import area_frame, fit_area_surface  # noqa: E402
from planning.follower import PathFollower  # noqa: E402
from planning.grid_planner import _nearest_free, resample  # noqa: E402
from planning.task_planner import PortalPlanner, patrol_waypoints  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402

DT = 0.002
DETECT_EVERY, FRAME_EVERY = 0.4, 0.08       # [s] sim time (video plays at 25 fps -> 2x speed)
F_DES = 5.0
DEPTH_MOUNT = CameraMount(pos=(0.30, 0.0, 0.05), pitch_deg=0.0)   # forward-looking: vertical surfaces seen face-on
DEPTH_HFOV, DEPTH_VFOV = 87.0, 58.0
PRE_DOCK = 0.12                             # [m] marker measured this far behind the stance (+0.12 m per retry)
MAX_REMAINDERS = 2                          # out-of-reach leftovers of one patch that are re-queued


def plate_normal(pts, p, n0, half=0.08, slab=0.025, min_pts=40):
    """Normal of the marker plate from depth points: the points within `half` of the marker centre (inside the
    20 cm black square) and `slab` of its PnP plane; SVD plane. None if too few points (not in the depth view)."""
    if pts is None or len(pts) == 0:
        return None
    rel = pts - p
    off = rel @ n0
    lat = np.linalg.norm(rel - np.outer(off, n0), axis=1)
    q = pts[(np.abs(off) < slab) & (lat < half)]
    if len(q) < min_pts:
        return None
    _, _, Vt = np.linalg.svd(q - q.mean(0))
    n = Vt[2] if Vt[2] @ n0 > 0 else -Vt[2]
    n = np.array([n[0], n[1], 0.0])                  # work surfaces (and their tags) are vertical
    return n / np.linalg.norm(n)


class PlanError(RuntimeError):
    pass


@dataclass
class Task:
    key: str
    mid: int
    kind: str                  # "survey" | "patch"
    pose: np.ndarray           # where to go (x, y, yaw), estimated frame (for ordering; recomputed before use)
    group: int
    info: dict = field(default_factory=dict)


class Mission:
    # what a mission variant overrides (sim/curved_mission.py): the world, its work orders, the printed marker size
    work_orders = WORK_ORDERS
    marker_size = MARKER_SIZE

    def make_world(self, args):
        return aruco_world(decimate=args.terrain_decimate, pipe_radius=tuple(args.pipe_radius), panel_seed=args.panel_seed)

    def __init__(self, args):
        self.args = args
        self.world = self.make_world(args)
        self.cfg = ROBOTS["go2_z1_official"]()
        self.m = build_model(self.cfg, torch_mass=0.0, world=self.world, contact_tip=True, scene_fn=add_head_camera,
                             render_size=(1280, 720))
        self.ctrl = controller_model(self.cfg, torch_mass=0.0)
        for mm in (self.m, self.ctrl):           # legs: the joint dynamics the RL policies were trained with
            for j in self.cfg.leg_joints:
                dof = mm.jnt_dofadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]
                mm.dof_damping[dof], mm.dof_frictionloss[dof] = 0.0, 0.01
        self.m.opt.timestep = DT
        # render near plane: MuJoCo's default (1 % of the scene extent) is 0.3 m on this 25 m scene - it clipped the
        # marker out of the head camera's view at the work stance (0.3 m away). A real camera sees it: 2 cm.
        self.m.vis.map.znear = 0.02 / self.m.stat.extent
        self.d = mujoco.MjData(self.m)
        self.ix = Indexer(self.m, self.cfg)
        x, y, yaw = self.world.start
        initial_state(self.m, self.d, self.ix, xy=(x, y), yaw=yaw)
        self.d.qpos[2] += self.world.ground_z(x, y) + 0.01
        mujoco.mj_forward(self.m, self.d)
        self.odo = Odometry((x, y), yaw, seed=args.seed)
        self.pol = BlindPolicy(self.m, self.cfg, args.policy)
        self.planner = PortalPlanner(self.world)
        self.rgb = ArucoCamera(self.m, self.marker_size)
        self.depth = DepthCamera(self.m, DEPTH_MOUNT, hfov_deg=DEPTH_HFOV, vfov_deg=DEPTH_VFOV, seed=args.seed)
        self.markers = MarkerMap(self.work_orders.keys())
        self.sensor = ForceSensor(self.m)
        self.pending, self.done, self.failed, self.queue, self.last_obs = {}, [], [], [], []
        self.reloc_on = True          # global landmark re-localisation (navigation); paused while docking on a marker
        self.events, self.walk_log, self.odo_log, self.path_now, self.cloud = [], [], [], None, None
        self.t_detect, self.t_frame = -1.0, -1.0
        self.canvases = self._make_canvases()
        self.summary = dict(tasks={}, surveys={}, markers={}, orders=[])
        self.video = None if args.no_video else Video(self.m, self.ix)
        self.patrol = patrol_waypoints(self.planner, (x, y))
        self.log(f"patrol: {len(self.patrol)} waypoints sampled from the map's free space")

    # ------------------------------------------------------------------ ground truth (drawing + scoring only)
    def _true_frame(self, mid):
        pos, R = self.world.true_marker(mid)
        wo = self.work_orders[mid]
        return area_frame(pos, R, wo["area_offset"], wo["area_size"])

    def _make_canvases(self):
        cv = {}
        for mid in self.work_orders:
            f, mg = self._true_frame(mid), 0.08
            cv[mid] = SurfaceCanvas(self.m, self.d, f.o - mg * f.ex - mg * f.ey, f.ex, f.ey, f.ez,
                                    (f.W + 2 * mg, f.H + 2 * mg), (mg, mg, mg + f.W, mg + f.H), cell=0.006,
                                    color=self.work_orders[mid]["color"])
        return cv

    def log(self, msg):
        self.events.append(dict(t=round(self.d.time, 2), msg=msg))
        print(f"[{self.d.time:6.1f}s] {msg}", flush=True)

    def pose(self):
        return self.odo.pose()

    # ------------------------------------------------------------------ perception: markers + re-localisation
    def detect(self, interrupt_ok=False):
        obs = self.rgb.detect(self.m, self.d)
        for o in obs:                                 # camera on the robot: relative pose -> estimated frame
            o.pos = self.odo.to_est(o.pos)
            o.R = self.odo.vec_to_est(o.R.T).T
        self.last_obs = obs
        new, reloc = self.markers.update(obs)
        for o, p, n in (reloc if self.reloc_on else []):   # ambiguous orientation: correct the position only
            n_obs = o.R[:2, 2] / np.linalg.norm(o.R[:2, 2])
            self.odo.correct(o.pos, n_obs, p, n if o.rot_ok else n_obs)
        for mid in new:
            self.log(f"marker {mid} confirmed at {np.round(self.markers.pose(mid)[0], 2)} (est. frame)")
            self._add_survey(mid)
        if self.video is not None:
            self.video.inset(self.rgb, obs)
        return bool(new)

    def frame_est(self, mid):
        p, R = self.markers.pose(mid)
        wo = self.work_orders[mid]
        return area_frame(p, R, wo["area_offset"], wo["area_size"])

    def fresh_frame(self, mid, T=1.0, fallback=None):
        """The area frame from this stance only: the marker measured relative to the robot over T s of standing
        (est. frame is constant while standing). The work is placed by this relative measurement - not by the
        global anchor, which carries the odometry drift of whenever the marker was first seen."""
        seen, clouds = [], []

        def tick():
            if self.d.time - self.t_detect >= DETECT_EVERY / 4:
                self.t_detect = self.d.time
                self.detect()
                seen.extend(o for o in self.last_obs if o.id == mid)
                if len(clouds) < 4:
                    clouds.append(self.depth.capture(self.m, self.d)[0])
        self.stand(T, on_tick=tick)
        if not seen:
            if fallback is not None and fallback is not False and clouds:
                return self.refine_on_plate(fallback, mid, self.odo.to_est(np.concatenate(clouds)))
            if fallback is not None:
                return fallback
            self.log(f"area {mid}: marker not in view from here - using its map anchor")
            return self.frame_est(mid)
        p, R = MarkerEstimate(mid, obs=seen).fused()
        n = plate_normal(self.odo.to_est(np.concatenate(clouds)) if clouds else None, p, R[:, 2])
        if n is not None:                             # the tag is a flat plate: its normal from the depth points on it
            R = np.stack([np.cross([0, 0, 1.0], n), [0, 0, 1.0], n], 1)
        elif not any(o.rot_ok for o in seen):         # face-on PnP ambiguity in every view: orientation from the anchor
            R = self.markers.pose(mid)[1]
        wo = self.work_orders[mid]
        return area_frame(p, R, wo["area_offset"], wo["area_size"])

    def refine_on_plate(self, f, mid, pts):
        """Marker too close for the RGB camera to decode, but the depth camera still sees its flat plate: re-measure the
        plate plane (normal + distance along it) from the depth points around where the frame `f` puts the marker.
        Removes the depth and normal error the last blind odometry step adds; lateral stays from `f`."""
        wo = self.work_orders[mid]
        p = f.o - (wo["area_offset"][0] - f.W / 2) * f.ex - (wo["area_offset"][1] - f.H / 2) * f.ey
        n = plate_normal(pts, p, f.ez)
        if n is None:
            return f
        e0 = np.array([*f.ez[:2], 0.0]) / np.linalg.norm(f.ez[:2])
        dth = float(np.arctan2(e0[0] * n[1] - e0[1] * n[0], e0 @ n))
        rel = pts - p
        off = rel @ n
        lat = np.linalg.norm(rel - np.outer(off, n), axis=1)
        sel = (np.abs(off) < 0.025) & (lat < 0.08)
        shift = float(np.median(off[sel]))
        if abs(shift) > 0.05:
            return f
        R = np.stack([np.cross([0, 0, 1.0], n), [0, 0, 1.0], n], 1)
        self.log(f"area {mid}: marker plate re-measured by depth at the stance: normal {np.degrees(dth):+.2f} deg, "
                 f"{1000 * shift:+.0f} mm along its normal")
        return area_frame(p + shift * n, R, wo["area_offset"], wo["area_size"])

    def survey_distance(self, f):
        """Base distance from the area at which the forward depth camera sees the whole area (+ 3-5 cm) face-on: as close
        as that allows - depth noise grows with distance^2 (1.25 m -> 1.10 m halves the normal error)."""
        cam_h = self.d.xpos[self.ix.base][2] + DEPTH_MOUNT.pos[2] - self.ground_under()
        top = (f.o + f.H * f.ey)[2] - self.ground_under() + 0.03
        need_h = (f.W / 2 + 0.05) / np.tan(np.radians(DEPTH_HFOV / 2))
        need_v = max(top - cam_h, 0.0) / np.tan(np.radians(DEPTH_VFOV / 2))
        return float(np.clip(max(need_h, need_v) + DEPTH_MOUNT.pos[0], 1.0, 2.2))

    def _add_survey(self, mid):
        f = self.frame_est(mid)
        D = self.survey_distance(f)
        c = f.o + f.W / 2 * f.ex
        xy = self.free_near(c[:2] + D * f.ez[:2])
        self.pending[f"{mid}"] = Task(f"{mid}", mid, "survey", np.array([*xy, np.arctan2(-f.ez[1], -f.ez[0])]), mid,
                                      info=dict(D_survey=D))
        p, _ = self.markers.pose(mid)
        self.planner.add_obstacle(("circle", p[0], p[1], 0.12))      # the marker is on *something*
        self.reorder()

    def free_near(self, xy):
        g = self.planner.grid
        if g.free[g.to_cell(xy)] or self.planner.on_platform(xy):   # the stair platform is a valid standing area
            return np.asarray(xy, float)
        return g.to_xy(_nearest_free(g, g.to_cell(xy)))

    def reorder(self):
        if not self.pending:
            self.queue = []
            return []
        xy, _ = self.pose()
        c0 = time.perf_counter()
        keys, cost = self.planner.order(xy, [(k, t.pose, t.group) for k, t in self.pending.items()])
        if keys is None:                              # nothing reachable on the map: try them in the given order
            keys = list(self.pending)
        self.queue = keys
        self.summary["orders"].append(dict(t=round(self.d.time, 2), queue=keys, cost_m=round(float(cost), 2),
                                           ms=round(1000 * (time.perf_counter() - c0), 1)))
        self.log(f"queue -> {keys} (travel cost {cost:.1f} m)")
        return keys

    def ground_under(self):
        """Height of the ground under the robot from its own leg kinematics (lowest foot)."""
        return float(self.d.geom_xpos[self.ix.feet, 2].min() - self.ix.foot_radius)

    # ------------------------------------------------------------------ walking
    def step_rl(self, cmd):
        self.pol.torques(self.m, self.d, cmd)
        mujoco.mj_step(self.m, self.d)
        R = self.d.xmat[self.ix.base]
        if self.pol.k % 10 == 0:
            still = np.linalg.norm(self.d.qvel[0:2]) < 0.002 and abs(self.d.qvel[5]) < 0.003   # truly static only: a slow
            # settle shuffle (~1 cm/s) must still be integrated - skipping it lost 2-3 deg and cm per dock
            self.odo.update(self.d, R, 10 * DT, still=still)
            self._stairs_landmark()
            e = self.odo.error()
            self.odo_log.append((self.d.time, e[0], e[1]))
        if R[8] < 0.5:
            raise RuntimeError(f"robot fell at t={self.d.time:.2f}s, x={self.d.qpos[0]:.2f}")
        return R.reshape(3, 3)

    def _stairs_landmark(self):
        """The staircase is on the map, so climbing it is a position measurement along it. When the lowest foot
        (leg kinematics) rises above 3/4 of the first riser, all four feet have just left the floor: the rear feet are
        at the first riser. Base position along the stairs = riser + the rear foot's offset from the base (also leg
        kinematics). Corrects the odometry's drift along the corridor every time the robot climbs."""
        st = self.world.stairs
        feet = self.d.geom_xpos[self.ix.feet]
        low = float(feet[:, 2].min() - self.ix.foot_radius)
        above = low > st.base_z + 0.75 * st.rise
        was = getattr(self, "_all_on_step", False)
        self._all_on_step = above
        xy, yaw = self.pose()
        if not above or was or not (st.x_start - 1.5 <= xy[0] <= st.x_end + 1.5):     # near the stairs (est. may be off)
            return
        R = self.d.xmat[self.ix.base].reshape(3, 3)
        h = np.array([R[0, 0], R[1, 0]]) / np.linalg.norm(R[:2, 0])          # heading (body x), from the IMU
        rel = (feet[:, :2] - self.d.qpos[0:2]) @ h                            # feet along the heading (kinematics)
        if h[0] > 0.7:                                                        # climbing from the west end
            x_base = st.x_start + self.ix.foot_radius - rel.min()
        elif h[0] < -0.7:                                                     # climbing from the east end
            x_base = st.x_end - self.ix.foot_radius + rel.min()
        else:
            return
        dx = x_base - self.odo.p[0]
        if abs(dx) < 1.5:
            self.odo.p[0] = x_base
            self.odo.corrections.append((abs(dx), 0.0))
            self.log(f"stairs landmark: first riser crossed -> position along the corridor corrected by {100 * dx:+.0f} cm")

    def walk(self, path, goal_yaw, label, tol=(0.05, 4.0), interruptible=True, t_max=90.0, v_max=0.5):
        """Follow `path` (estimated frame) with the RL policy. 'arrived' | 'interrupted' (queue head changed)."""
        self.path_now = path
        st = self.planner
        fol = PathFollower(path, goal_yaw, v_max=v_max, vy_max=0.3, wz_max=0.8, dock_radius=0.8)
        t0, t_near, cmd = self.d.time, None, np.zeros(3)
        head = self.queue[0] if self.queue else None
        while self.d.time - t0 < t_max:
            xy, yaw = self.pose()
            if self.pol.k % 10 == 0:
                cmd = fol.command(xy, yaw)
                if st.on_stairs(xy) and not st.on_platform(xy, margin=0.0):   # on the steps: climb straight, slower
                    cmd[0] = min(cmd[0], 0.35); cmd[1] = 0.0
                near = np.linalg.norm(fol.goal - xy) < tol[0] and abs(wrap(goal_yaw - yaw)) < np.radians(tol[1])
                t_near = (t_near or self.d.time) if near else None
                if fol.done or (t_near is not None and self.d.time - t_near > 0.6):
                    break
            R = self.step_rl(cmd)
            self.walk_log.append((self.d.time, self.d.qpos[0], self.d.qpos[1],
                                  float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), label, xy[0], xy[1]))
            if self.d.time - self.t_detect >= DETECT_EVERY:
                self.t_detect = self.d.time
                if self.detect() and interruptible and self.args.explore and not st.on_stairs(xy):
                    if self.queue and self.queue[0] != head:
                        self.log(f"queue head changed {head} -> {self.queue[0]}: re-planning")
                        return "interrupted"
            self.frame(label, cmd)
        self.stand(0.6)
        return "arrived"

    def stand(self, T, detect=False, on_tick=None):
        t0 = self.d.time
        while self.d.time - t0 < T:
            self.step_rl(np.zeros(3))
            if on_tick is not None and self.pol.k % 10 == 0:
                on_tick()
            if detect and self.d.time - self.t_detect >= DETECT_EVERY / 2:
                self.t_detect = self.d.time
                self.detect()
            self.frame("standing", np.zeros(3))

    def goto(self, pose, label, approach=0.8, **kw):
        xy, _ = self.pose()
        c0 = time.perf_counter()
        try:
            path, info = self.planner.plan(xy, pose[:2], pose[2], approach=approach)
        except RuntimeError:
            try:                                      # straight approach blocked on the map: go to the nearest free point
                path, info = self.planner.plan(xy, self.free_near(pose[:2]), pose[2], approach=0.0)
            except RuntimeError as ex:
                raise PlanError(str(ex)) from None
        self.log(f"A* to {label}: {info['length']:.1f} m{' via stairs' if info['stairs'] else ''} "
                 f"({1000 * (time.perf_counter() - c0):.0f} ms)")
        return self.walk(path, pose[2], label, **kw)

    def go_pose(self, pose, label, approach=0.6, **kw):
        """goto() that also handles the stair platform as a standing area: climb along the portal to the platform
        centre, then walk on the flat top to the pose; leaving: back to the centre, then down the portal."""
        P = self.planner
        xy, _ = self.pose()
        goal_on = P.on_platform(pose[:2])
        if P.on_stairs(xy) and not (goal_on and P.on_platform(xy)):
            self.leave_platform(pose[:2])
        if not goal_on:
            return self.goto(pose, label, approach=approach, **kw)
        xy, _ = self.pose()
        if not P.on_platform(xy):
            a, b = P.portal if P.side(xy) == 0 else P.portal[::-1]
            yaw_in = float(np.arctan2(*(b - a)[::-1]))
            if self.goto(np.array([*a, yaw_in]), f"{label}: to the stairs", approach=0.6, **kw) == "interrupted":
                return "interrupted"
            self.line_to(np.array([*P.platform_center(), yaw_in]), f"{label}: climb to the platform", tol=(0.08, 6.0))
        return self.line_to(pose, f"{label}: on the platform", tol=(0.045, 2.5))

    def leave_platform(self, goal_xy):
        P = self.planner
        a, b = P.portal if P.side(goal_xy) == 1 else P.portal[::-1]     # walk towards the goal's side
        yaw_out = float(np.arctan2(*(b - a)[::-1]))
        self.line_to(np.array([*P.platform_center(), yaw_out]), "back to the platform centre", tol=(0.08, 6.0))
        self.line_to(np.array([*b, yaw_out]), "down the stairs", tol=(0.15, 10.0))

    def floor_z(self):
        xy, _ = self.pose()
        if not self.planner.on_stairs(xy):
            self._floor = self.ground_under()
        return getattr(self, "_floor", 0.0)

    def line_to(self, pose, label, tol=(0.03, 2.5)):
        xy, _ = self.pose()
        return self.walk(resample(np.array([xy, pose[:2]]), 0.05), pose[2], label, tol=tol, interruptible=False, t_max=20)

    def look_around(self, label="look around"):
        for wz, T in [(0.6, 1.8), (-0.6, 3.6), (0.6, 1.8)]:
            t0 = self.d.time
            while self.d.time - t0 < T:
                self.step_rl(np.array([0.0, 0.0, wz]))
                if self.d.time - self.t_detect >= DETECT_EVERY / 2:
                    self.t_detect = self.d.time
                    self.detect()
                self.frame(label, np.array([0, 0, wz]))
        self.stand(0.5)

    # ------------------------------------------------------------------ tasks
    def run_task(self, task):
        try:
            return self.survey(task) if task.kind == "survey" else self.do_patch(task)
        finally:
            self.reloc_on = True

    def survey(self, task):
        mid = task.mid
        if self.go_pose(task.pose, f"area {mid}: survey pose", approach=0.6) == "interrupted":
            return "interrupted"
        f = self.fresh_frame(mid, 1.2, fallback=False)      # marker relative to the robot, from here
        if f is False:                                      # not in view (the anchor drifted): step closer, look again
            f0 = self.frame_est(mid)
            closer = np.array([*(np.asarray(task.pose[:2]) - 0.4 * f0.ez[:2]), np.arctan2(-f0.ez[1], -f0.ez[0])])
            self.line_to(closer, f"area {mid}: closer to find the marker")
            f = self.fresh_frame(mid, 1.2, fallback=False)
            if f is False:
                self.log(f"area {mid}: marker not in view at the survey pose - survey skipped")
                return "failed"

        D = self.survey_distance(f)
        pose = np.array([*self.free_near((f.o + f.W / 2 * f.ex + D * f.ez)[:2]), np.arctan2(-f.ez[1], -f.ez[0])])
        xy, yaw = self.pose()
        if np.linalg.norm(xy - pose[:2]) > 0.08 or abs(wrap(yaw - pose[2])) > np.radians(4):
            self.line_to(pose, f"area {mid}: face the area")
            f = self.fresh_frame(mid, 1.0, fallback=f)
        clouds = []
        self.stand(1.2, on_tick=lambda: clouds.append(self.depth.capture(self.m, self.d)[0]) if len(clouds) < 16 else None)
        pts = self.odo.to_est(np.concatenate(clouds))
        try:
            surf = fit_area_surface(pts, f)
        except RuntimeError as ex:
            self.log(f"area {mid}: {ex}")
            return "failed"
        q = f.to_area(pts)
        self.cloud = (pts[(q[:, 0] > -0.05) & (q[:, 0] < f.W + 0.05) & (q[:, 1] > -0.05) & (q[:, 1] < f.H + 0.05)
                          & (np.abs(q[:, 2]) < 0.3)][::2], [0.2, 1.0, 0.3, 1])
        acc = self.model_accuracy(mid, f, surf)
        self.log(f"area {mid}: surface model {surf.info['cells']} cells, depth range {surf.info['depth_range_mm'][0]:.0f}"
                 f"..{surf.info['depth_range_mm'][1]:.0f} mm; vs real surface: depth err {acc['depth_rms_mm']:.1f} mm RMS,"
                 f" normal err {acc['normal_rms_deg']:.1f} deg RMS")
        # what was seen is an obstacle (+ an assumed 20 cm of body behind the visible face)
        w = surf.w(*np.meshgrid(np.linspace(0, f.W, 9), np.linspace(0, f.H, 3)))
        w0, w1 = float(w.min()) - 0.20, float(w.max())
        c = f.to_world(f.W / 2, 0.0, (w0 + w1) / 2)
        self.planner.add_obstacle(("box", c[0], c[1], f.W / 2 + 0.02, (w1 - w0) / 2, float(np.arctan2(f.ex[1], f.ex[0]))))
        c0 = time.perf_counter()
        try:
            fz = self.floor_z()
            plan = plan_stance(self.cfg, self.ctrl, f, surf, lambda xy, yaw: self.planner.stance_ground(xy, yaw, fz))
        except RuntimeError as ex:
            self.log(f"area {mid}: {ex}")
            return "failed"
        self.log(f"area {mid}: reach plan ({time.perf_counter() - c0:.1f} s): standoff {plan.D:.2f} m, reachable "
                 f"{1000 * (plan.rel[1] - plan.rel[0]):.0f} mm wide -> {len(plan.patches)} patch(es) "
                 f"{[round(b - a, 3) for a, b, _ in plan.patches]} m")
        self.summary["surveys"][str(mid)] = dict(model=surf.info, accuracy=acc, standoff=plan.D, reach_table=plan.table,
                                                 patches=[(a, b) for a, b, _ in plan.patches])
        self.pending.pop(task.key, None)
        for i, (a, b, s) in enumerate(plan.patches):
            self.pending[f"{mid}.{i}"] = Task(f"{mid}.{i}", mid, "patch", stance_pose(f, plan.w_ref, plan.D, s), mid,
                                              info=dict(surf=surf, plan=plan, u0=a, u1=b, s=s, left=MAX_REMAINDERS))
        self.reorder()
        return "queued"

    def model_accuracy(self, mid, f_est, surf):
        """Fitted model vs the real surface (canvas ray-cast), seen through the robot's belief of where it is."""
        cv = self.canvases[mid]
        f = f_est.mapped(self.odo.to_true, self.odo.vec_to_true)
        P, N = cv.P[cv.target & cv.hit], cv.N[cv.target & cv.hit]
        q = f.to_area(P)
        dw = q[:, 2] - surf.w(q[:, 0], q[:, 1])
        nu = surf.normal_uvw(q[:, 0], q[:, 1])
        nm = nu[:, :1] * f.ex + nu[:, 1:2] * f.ey + nu[:, 2:3] * f.ez
        ang = np.degrees(np.arccos(np.clip(np.sum(nm * N, 1), -1, 1)))
        return dict(depth_rms_mm=float(1000 * np.sqrt(np.mean(dw ** 2))), depth_max_mm=float(1000 * np.abs(dw).max()),
                    normal_rms_deg=float(np.sqrt(np.mean(ang ** 2))), normal_max_deg=float(ang.max()))

    def do_patch(self, task):
        mid, I = task.mid, task.info
        plan, surf = I["plan"], I["surf"]
        dock = stance_pose(self.frame_est(mid), plan.w_ref, plan.D, I["s"])
        xy, _ = self.pose()
        if np.linalg.norm(xy - dock[:2]) < 1.2:
            self.line_to(dock, f"area {task.key}: sidestep")
        elif self.go_pose(dock, f"area {task.key}", approach=0.6) == "interrupted":
            return "interrupted"
        # fine dock on the marker. At the stance the marker is ~0.3 m from the head camera and barely fits in its
        # view, so it is measured from a pre-dock pose PRE_DOCK further back (relative to the robot, standing), then
        # the last short straight step runs on odometry (~1 cm drift). At the stance the marker is tried again.
        pose_of = lambda f: stance_pose(f, plan.w_ref, plan.D, I["s"])
        f_meas = None
        # from here to the hand-off the work is placed by the local, robot-relative marker measurement: global
        # re-localisation (pulling the estimate to the marker's stored map pose) would move the estimate under it
        self.reloc_on = False
        for k in range(3):
            self.settle()
            pre = pose_of(f_meas or self.frame_est(mid))
            pre[:2] += (PRE_DOCK + 0.12 * k) * np.array([-np.cos(pre[2]), -np.sin(pre[2])])   # not seen: back off more
            xy, yaw = self.pose()
            if np.linalg.norm(xy - pre[:2]) > 0.045 or abs(wrap(yaw - pre[2])) > np.radians(3.0):
                self.line_to(pre, f"area {task.key}: pre-dock", tol=(0.045, 2.5))
                self.settle()
            f_new = self.fresh_frame(mid, 1.0, fallback=False)
            if f_new is not False:
                f_meas = f_new
                break
            self.log(f"area {task.key}: marker not in view from the pre-dock pose")
        if f_meas is None:
            self.log(f"area {task.key}: no marker measurement - using its map anchor")
            f_meas = self.frame_est(mid)
        dock = pose_of(f_meas)
        self.line_to(dock, f"area {task.key}: dock", tol=(0.045, 2.5))
        self.settle()
        task.pose = dock
        # the patch in the physics frame (the WBC's frame): the robot's measured geometry, moved by its pose belief
        f_est = self.fresh_frame(mid, 1.0, fallback=f_meas)
        xy, yaw = self.pose()
        self.log(f"area {task.key}: at the stance, {1000 * np.linalg.norm(xy - pose_of(f_est)[:2]):.0f} mm / "
                 f"{np.degrees(abs(wrap(yaw - pose_of(f_est)[2]))):.1f} deg from the marker-relative dock")
        patch = SurfacePatch(f_est, surf, I["u0"], I["u1"]).mapped(self.odo.to_true, self.odo.vec_to_true)
        sp, _ = reachable_span(self.cfg, self.ctrl, self.d.qpos.copy(), patch)
        if sp is None or sp[1] - sp[0] < 0.03:
            self.log(f"area {task.key}: nothing reachable from this stance")
            return "failed"
        rest = [(a, b) for a, b in ((I["u0"], sp[0]), (sp[1], I["u1"])) if b - a > 0.03]   # < 3 cm: left uncovered (reported by the paint coverage)
        if rest:
            self.log(f"area {task.key}: reachable u {sp[0]:.3f}..{sp[1]:.3f} of {I['u0']:.3f}..{I['u1']:.3f} m")
            if I["left"] > 0:
                lo, hi = plan.rel
                for j, (a, b) in enumerate(rest):
                    key = f"{task.key}r{j}"
                    s = (a + b) / 2 - (lo + hi) / 2
                    self.pending[key] = Task(key, mid, "patch", stance_pose(f_est, plan.w_ref, plan.D, s), mid,
                                             info=dict(I, u0=a, u1=b, s=s, left=I["left"] - 1))
                    self.log(f"area {task.key}: out-of-reach rest {b - a:.3f} m queued as {key}")
            patch = patch.sub(*sp)
        # true dock error w.r.t. the real marker (what the marker re-localisation is for)
        intended = stance_pose(self._true_frame(mid), plan.w_ref, plan.D, I["s"])
        R = self.d.xmat[self.ix.base].reshape(3, 3)
        dock_err = (1000 * float(np.linalg.norm(self.d.qpos[:2] - intended[:2])),
                    float(np.degrees(abs(wrap(np.arctan2(R[1, 0], R[0, 0]) - intended[2])))))
        return self.trace(task, patch, dock_err)

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
            t_ok = (t_ok or self.d.time) if (feet <= touching and still) else None
            if t_ok is not None and self.d.time - t_ok > 0.3:
                return True
        return False

    def trace(self, task, area, dock_err):
        """Hand-off RL -> stance WBC (base) + arm admittance; trace the area; hand back to RL."""
        mid = task.mid
        cv = self.canvases[mid]
        try:
            at = AreaTask(self.cfg, self.ctrl, self.d.qpos.copy(), area, t0=self.d.time, f_des=F_DES)
        except RuntimeError as ex:
            self.log(f"area {task.key}: {ex}")
            return "failed"
        wbc = StanceWBC(self.ctrl, self.cfg, **dict(self.cfg.wbc_gains, **TRACE_WBC_GAINS))
        self.sensor = ForceSensor(self.m)
        base0 = self.d.qpos[0:3].copy()
        F, drift = [], 0.0
        oe = self.odo.error()
        self.log(f"area {task.key}: hand-off, true dock error {dock_err[0]:.0f} mm / {dock_err[1]:.1f} deg "
                 f"(odometry vs truth {1000 * oe[0]:.0f} mm / {oe[1]:.1f} deg); "
                 f"{area.width:.2f} x {area.height:.2f} m, {len(at.segs)} raster segments, IK margin {at.ik_margin:.2f}, "
                 f"{at.T:.0f} s")
        while self.d.time < at.t0 + at.T:
            f = self.sensor.read(self.m, self.d)
            ref = at.ref(self.d.time, self.d.qpos.copy(), f)
            tau, _ = wbc.compute(self.d.qpos.copy(), self.d.qvel.copy(), ref)
            self.d.ctrl[:] = tau[self.ix.act_dof]
            mujoco.mj_step(self.m, self.d)
            ph = at.phase(self.d.time)
            cv.stamp(self.d.site_xpos[self.ix.tip].copy(), self.sensor.raw, F_DES)
            if self.d.xmat[self.ix.base][8] < 0.85:
                raise RuntimeError(f"robot tipped over during {task.key} (phase {ph}, t={self.d.time:.2f}s)")
            if ph == "raster":
                F.append(self.sensor.raw)
                drift = max(drift, float(np.linalg.norm(self.d.qpos[0:2] - base0[0:2])))
            self.frame(f"work {task.key}: {ph}", None, work=area, force=f, prog=at.progress(self.d.time), canvas=cv)
        self.pol.reset()
        F = np.array(F)
        res = dict(u_range=[area.u0, area.u1], force_mean=float(F.mean()),
                   force_rms_err=float(np.sqrt(np.mean((F - F_DES) ** 2))),
                   force_p5_p95=[float(np.percentile(F, 5)), float(np.percentile(F, 95))],
                   contact_lost=float(np.mean(F < 0.5)), base_drift_mm=1000 * drift, wbc_qp_failures=int(wbc.fail),
                   ik_margin=at.ik_margin, approach_wait_s=round(at.wait_s, 2), true_dock_err_mm_deg=dock_err)
        self.summary["tasks"][task.key] = res
        ok = res["contact_lost"] < 0.5
        self.log(f"area {task.key} {'done' if ok else 'FAILED (no contact)'}: F {res['force_mean']:.2f} N "
                 f"(rms err {res['force_rms_err']:.2f}, p5-p95 {res['force_p5_p95'][0]:.1f}-{res['force_p5_p95'][1]:.1f}),"
                 f" contact lost {100 * res['contact_lost']:.1f} %, base drift {res['base_drift_mm']:.1f} mm")
        return "done" if ok else "failed"

    # ------------------------------------------------------------------ main loop
    def run_task_safe(self, task):
        try:
            r = self.run_task(task)
        except PlanError as ex:
            self.log(f"{task.key}: no path ({ex})")
            r = "failed"
        if r in ("done", "failed"):
            self.pending.pop(task.key, None)
            (self.done if r == "done" else self.failed).append(task.key)
        return r

    def run_sites(self):
        """Fixed sites (no exploration): visit SITE_ORDER at the operator's rough viewpoints; at each, find the marker
        (look around if needed), then survey + all its patches before moving on."""
        self.log(f"start (fixed sites {SITE_ORDER}): map known, marker poses + surfaces unknown")
        for mid in SITE_ORDER:
            hint = np.array(SITE_HINTS[mid], float)
            try:
                self.go_pose(hint, f"site {mid}", approach=0.6)
            except PlanError as ex:
                self.log(f"site {mid}: no path ({ex})")
                self.failed.append(str(mid))
                continue
            self.stand(1.0, detect=True)
            if mid not in self.markers.confirmed:
                self.look_around(f"site {mid}: look for the marker")
            if mid not in self.markers.confirmed:
                self.log(f"site {mid}: marker not found")
                self.failed.append(str(mid))
                continue
            while True:
                mine = sorted(k for k, t in self.pending.items() if t.mid == mid)
                if not mine:
                    break
                self.queue = mine
                self.run_task_safe(self.pending[mine[0]])
        self.log(f"mission complete: done {self.done}, failed {self.failed}")
        self.stand(1.5)

    def run(self):
        if not self.args.explore:
            return self.run_sites()
        self.log(f"start: work orders {sorted(self.work_orders)}, map known, marker positions + surfaces unknown")
        self.look_around()
        while True:
            if self.pending:
                if not self.queue or self.queue[0] not in self.pending:
                    self.reorder()
                task = self.pending[self.queue[0]]
                try:
                    r = self.run_task(task)
                except PlanError as ex:
                    self.log(f"{task.key}: no path ({ex})")
                    r = "failed"
                if r in ("done", "failed"):
                    self.pending.pop(task.key, None)
                    (self.done if r == "done" else self.failed).append(task.key)
                    self.reorder()
                continue
            if len(self.markers.confirmed) == len(self.work_orders) or not self.patrol:
                break
            w = self.patrol.pop(0)
            xy, _ = self.pose()
            if np.linalg.norm(w - xy) < 1.0:
                continue
            try:
                self.goto(np.array([*w, np.arctan2(*(w - xy)[::-1])]), "patrol", approach=0.0, tol=(0.25, 25.0))
            except PlanError as ex:
                self.log(f"patrol waypoint {np.round(w, 1)} skipped: {ex}")
                continue
            if not self.pending:
                self.look_around("patrol: look around")
        self.log(f"mission complete: done {self.done}, failed {self.failed}")
        self.stand(1.5)

    # ------------------------------------------------------------------ video
    def frame(self, label, cmd, work=None, force=None, prog=None, canvas=None):
        if self.video is None or self.d.time - self.t_frame < FRAME_EVERY - 1e-9:
            return
        self.t_frame = self.d.time
        q = " > ".join(self.queue[:6]) if self.queue else "-"
        e = self.odo.error()
        l1 = f"t={self.d.time:6.1f}s  {label}   odometry error {100 * e[0]:.0f} cm / {e[1]:.1f} deg"
        if force is not None:
            l1 += f"   F = {force:4.1f} N (target {F_DES:.0f})  {100 * prog:3.0f} %"
        l2 = f"queue: {q}   done: {', '.join(self.done) or '-'}   markers: {self.markers.confirmed}"
        self.video.frame(self.m, self.d, [l1, l2], self.canvases.values(), self.path_now, work,
                         self.cloud if work is None else None, canvas, self.odo)


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

    def frame(self, m, d, lines, canvases, path, work, cloud, canvas, odo):
        from PIL import Image, ImageDraw
        R = d.xmat[self.ix.base].reshape(3, 3)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        if work is not None:
            c, _ = work.point(work.width / 2, work.height / 2)
            self.work.lookat[:] = c
            self.work.distance, self.work.azimuth, self.work.elevation = 1.4, yaw + 35, -12
            cam = self.work
        else:
            self.track.azimuth = yaw + 25
            cam = self.track
        self.r.update_scene(d, camera=cam)
        scn = self.r.scene
        for cv in canvases:
            cv.draw(scn)
        if path is not None and work is None:            # planned path (estimated frame) drawn where the robot
            P = odo.to_true(np.c_[path[::5], np.full(len(path[::5]), 0.11)])     # believes it is
            for p in P:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], p,
                                    np.eye(3).ravel(), np.array([0.1, 0.5, 1.0, 0.8], np.float32))
                scn.ngeom += 1
        if cloud is not None:
            pts, rgba = cloud
            for p in odo.to_true(pts[:: max(1, len(pts) // 1500)]):
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], p,
                                    np.eye(3).ravel(), np.array(rgba, np.float32))
                scn.ngeom += 1
        im = self.r.render().copy()
        if self.inset_img is not None:
            im[8:188, -328:-8] = self.inset_img
        pil = Image.fromarray(im)
        dr = ImageDraw.Draw(pil)
        if canvas is not None:                               # intensity-brush legend: shade = contact force
            x0, y0, wbar = 12, 12, 220
            for k in range(wbar):
                F = 2 * F_DES * k / (wbar - 1)
                dr.line([x0 + k, y0, x0 + k, y0 + 14], fill=tuple(int(255 * c) for c in canvas.shade(F)))
            dr.rectangle([x0 - 1, y0 - 1, x0 + wbar, y0 + 15], outline=(255, 255, 255))
            for F in (0, F_DES, 2 * F_DES):
                xx = x0 + int((wbar - 1) * F / (2 * F_DES))
                dr.text((xx - 6, y0 + 17), f"{F:.0f} N", fill=(255, 255, 255))
            dr.text((x0, y0 + 30), "brush shade = measured contact force", fill=(255, 255, 255))
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
    tr = np.array([(r[1], r[2]) for r in ms.walk_log])
    est = np.array([(r[5], r[6]) for r in ms.walk_log])
    if len(tr):
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#1f6feb", lw=1.2, label="walked (true)")
        ax.plot(est[:, 0], est[:, 1], ":", color="#cf222e", lw=1.0, label="walked (robot's estimate)")
    for mid in ms.work_orders:
        pos, _ = w.true_marker(mid)
        ax.plot(pos[0], pos[1], "ks", ms=6)
        ax.annotate(f" {mid}", pos[:2], fontsize=8)
    for t in w.tasks:
        if t.pipe is not None:
            ax.add_patch(plt.Circle(t.pipe.xy, t.pipe.radius, color="k", fill=False))
    ax.plot([], [], "ks", label="marker (true)")
    ax.plot(*w.start[:2], "go", ms=8, label="start")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Known map (grey: obstacles + robot-radius margin) - walked path, true vs estimated")
    fig.tight_layout(); fig.savefig(path_png, dpi=110); plt.close(fig)


def plot_paint(ms, path_png):
    """Force map of every work area (intensity brush, in N) on the real surface's (u, v) grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cvs = list(ms.canvases.items())
    fig, axs = plt.subplots(1, len(cvs), figsize=(4.2 * len(cvs), 3.0), squeeze=False)
    for ax, (mid, cv) in zip(axs[0], cvs):
        ni, nj = cv.F.shape
        im = ax.imshow(cv.F.T, origin="lower", extent=[0, ni * cv.cell, 0, nj * cv.cell], cmap="viridis",
                       vmin=0, vmax=2 * F_DES)
        u = np.nonzero(cv.target.any(1))[0]; v = np.nonzero(cv.target.any(0))[0]
        ax.add_patch(plt.Rectangle((u[0] * cv.cell, v[0] * cv.cell), (u[-1] - u[0] + 1) * cv.cell,
                                   (v[-1] - v[0] + 1) * cv.cell, fill=False, ec="r", lw=1, ls="--"))
        mt = cv.metrics()
        w20 = "" if mt["within_20pct"] is None else f", {100 * mt['within_20pct']:.0f} % within 5 N +-20 %"
        ax.set_title(f"marker {mid}: coverage {100 * mt['coverage']:.0f} %{w20}", fontsize=8)
        ax.set_xlabel("u [m]"); ax.set_ylabel("v [m]")
    fig.colorbar(im, ax=axs[0].tolist(), label="contact force [N]", shrink=0.8)
    fig.savefig(path_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="moe_cts", choices=["moe_cts", "symmetry"])
    ap.add_argument("--terrain-decimate", type=int, default=1)
    ap.add_argument("--panel-seed", type=int, default=4)
    ap.add_argument("--pipe-radius", type=float, nargs=2, default=[0.11, 0.13])
    ap.add_argument("--seed", type=int, default=1, help="odometry + depth noise seed")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--explore", action="store_true", help="patrol + find markers (else: fixed sites SITE_ORDER)")
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
    oe = np.array([(r[1], r[2]) for r in ms.odo_log]) if ms.odo_log else np.zeros((1, 2))
    S["paint"] = {str(k): v.metrics() for k, v in ms.canvases.items()}
    S["localisation"] = dict(odometry_err_max_m=float(oe[:, 0].max()), odometry_err_final_m=float(oe[-1, 0]),
                             heading_err_max_deg=float(np.abs(oe[:, 1]).max()), marker_corrections=len(ms.odo.corrections))
    S.update(done=ms.done, failed=ms.failed, aborted=err, sim_time_s=float(ms.d.time), wall_time_s=time.time() - c0,
             walking=dict(max_tilt_deg=float(tilt.max()), distance_m=float(np.sum(np.linalg.norm(np.diff(
                 np.array([(r[1], r[2]) for r in ms.walk_log]), axis=0), axis=1))) if len(ms.walk_log) > 1 else 0.0),
             events=ms.events, policy=args.policy, panel_seed=args.panel_seed, pipe_radius=args.pipe_radius)
    json.dump(S, open(os.path.join(out, f"summary{args.tag}.json"), "w"), indent=2, default=float)
    plot_map(ms, os.path.join(out, f"map{args.tag}.png"))
    plot_paint(ms, os.path.join(out, f"paint{args.tag}.png"))
    if ms.video is not None and ms.video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission{args.tag}.mp4"), ms.video.frames, fps=25, quality=7, macro_block_size=1)
    print(json.dumps({k: S[k] for k in ("done", "failed", "aborted", "paint", "localisation", "surveys")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
