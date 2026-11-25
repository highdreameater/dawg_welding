"""ArUco mission: find and visit four ArUco markers stuck on curved surfaces, one after the other, autonomously.

    python sim/aruco_mission.py                        # Go2 + Z1, fine-tuned walking policy -> results/aruco/
    python sim/aruco_mission.py --no-video --seed 1
    python sim/aruco_mission.py --loco go2             # pretrained Isaac Lab Go2 policy

World: sim/world.py aruco_workshop() - the workshop map (walls, obstacles, stations) + a staircase + four curved
surfaces with one marker each (column, vessel, spherical tank, pipe). The map is known (no SLAM; the robot's pose
comes from the simulator); the markers are not in the map.

Loop (state machine, all decisions from the robot's own detections):
    OBSERVE   turn 360 deg in place (RL policy), head RGB camera every 0.25 s -> ArUco detection + PnP pose ->
              registry (fused estimate per id) + explorer (which surfaces have been seen)
    SELECT    nearest detected, unvisited marker; none -> EXPLORE
    NAVIGATE  approach pose in front of the marker (standoff along its normal, facing it), A* on the map
              (obstacles + stairs not traversable), path follower -> RL policy. Detections keep refining the
              marker estimate on the way; if the approach pose moves by > 10 cm / 6 deg, replan.
    CONFIRM   hand-off to the stance WBC: level, brace, hold rigid; the camera re-measures the marker; confirmed if
              it is seen and the robot stands within tolerance in front of it (lateral, distance, heading) with the
              base still. Otherwise re-dock (at most 2 more times). Hand back to the RL policy.
    VISITED   recorded in the registry (never selected again) -> OBSERVE again at the target
    EXPLORE   walk to the viewpoint that reveals the most unseen curved surface (sim/planning/explorer.py), abort
              as soon as a new marker is detected, then OBSERVE
    DONE      all four markers visited -> "TASK COMPLETE"; or no viewpoint left / time limit -> "TASK INCOMPLETE"

Controllers (unchanged): legs walk with the RL velocity policy (sim/locomotion/policy.py, arm stowed by joint PD),
standing at a target is the stance WBC (sim/wbc/stance_wbc.py, references from sim/stance_hold.py). The arm's
admittance (force) task is not used: the markers are visited, not touched.
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locomotion.policy import LOCO, ROBOT_LOCO, SIM_DT as RL_DT, VelocityPolicy  # noqa: E402
from markers import marker_frame  # noqa: E402
from mission_demo import FPS, Video, base_pose, world_collisions  # noqa: E402
from perception.aruco import ArucoDetector, RgbCamera, camera_spec, draw_detections  # noqa: E402
from planning.explorer import Explorer  # noqa: E402
from planning.follower import PathFollower, wrap  # noqa: E402
from planning.grid_planner import OccupancyGrid, plan_path  # noqa: E402
from planning.targets import MarkerRegistry, approach_poses, plan_to_marker  # noqa: E402
from stance_hold import StanceHold  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import WORLDS, Cylinder, Ellipsoid, Pipe  # noqa: E402

WBC_DT = 0.002
CAPTURE_DT = 0.25          # camera frame period while walking [s]
OBSERVE_DT = 0.125         # ... while turning in place to observe (~5 deg per frame) [s]
# alignment accepted at a target (robot belief: fused marker estimate)
ALIGN_TOL = dict(lateral_m=0.06, distance_m=0.10, heading_deg=10.0, base_speed_mm_s=10.0)


class MissionVideo(Video):
    """mission_demo.Video (tracking camera, sphere markers) + the head camera image with detections as an inset."""

    def __init__(self, m, ix, speed):
        super().__init__(m, ix, dict(lookat=[0, 0, 0], distance=3.0, azimuth=0, elevation=-30))
        self.track.distance, self.track.elevation = 3.4, -55
        self.speed, self.inset = speed, None

    def due(self, t):
        return t - self.last >= self.speed / FPS - 1e-9

    def frame(self, d, text, cam="track"):
        super().frame(d, text, cam)
        if self.inset is not None:
            f = self.frames[-1].copy()
            h, w = self.inset.shape[:2]
            f[44:44 + h, -w - 6:-6] = self.inset
            self.frames[-1] = f


class ArucoMission:
    def __init__(self, args):
        import cv2
        self.cv2, self.args = cv2, args
        self.cfg = cfg = ROBOTS[args.robot]()
        self.world = WORLDS[args.world]()
        self.plant = build_model(cfg, world=self.world, contact_tip=True, cameras=[camera_spec(cfg.name)])
        self.ctrl = controller_model(cfg)
        self.d = mujoco.MjData(self.plant)
        self.ix = Indexer(self.plant, cfg)
        initial_state(self.plant, self.d, self.ix, xy=self.world.start[:2], yaw=self.world.start[2])
        loco = args.loco
        if loco == "auto":
            loco = ROBOT_LOCO[args.robot]
            if args.robot == "go2_z1" and os.path.exists(LOCO["go2_payload"].checkpoint):
                loco = "go2_payload"
        self.loco = loco
        self.pol = VelocityPolicy(self.plant, cfg, loco=loco)
        self.grid = OccupancyGrid(self.world)
        self.cam = RgbCamera(self.plant, noise_std=args.pixel_noise, seed=args.seed)
        self.det = ArucoDetector(self.cam.K, marker_size=args.marker_size)
        # a marker is stuck on an obstacle surface: its estimate must lie within 15 cm of an occupied map cell
        self.reg = MarkerRegistry(n_expected=len(self.world.markers), min_views=3,
                                  plausible=lambda p: self.grid.clearance_at(p[:2]) <= 0.15)
        self.explorer = Explorer(self.grid, self.world, range_m=args.range)
        self.video = None if args.no_video else MissionVideo(self.plant, self.ix, args.video_speed)
        self.truth = {mk.id: dict(zip(("p", "n"), marker_frame(self.world, mk)[:2]), surface=mk.surface)
                      for mk in self.world.markers}
        self.stairs_geoms = {i for i in range(self.plant.ngeom)
                             if (mujoco.mj_id2name(self.plant, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("stairs")}
        self.foot_geoms = set(self.ix.feet.tolist())
        self.log, self.events, self.paths, self.viewpoints = [], [], [], []
        self.state, self.last_capture, self.first_detection = "init", -1.0, {}
        self.stairs_contact, self.revisits, self.verifications = False, 0, 0

    # ------------------------------------------------------------------ helpers
    def xy_yaw(self):
        return base_pose(self.d, self.ix)

    def event(self, kind, **kw):
        e = dict(t=round(float(self.d.time), 3), event=kind, **kw)
        self.events.append(e)
        print(f"[t={self.d.time:6.1f}s] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    def status_line(self):
        vis = self.reg.order_visited
        det = [i for i in self.reg.order_detected if i not in vis]
        return f"visited {vis}  detected {det}  ({len(vis)}/{self.reg.n_expected})"

    def capture(self):
        """One camera frame: detect, fuse, update surface coverage. Returns (observations, newly detected ids)."""
        self.last_capture = self.d.time
        img, cp, cR = self.cam.capture(self.plant, self.d)
        obs = self.det.detect(img, cp, cR, t=self.d.time)
        new = self.reg.update(obs)
        fwd = -cR[:, 2]
        self.explorer.mark_seen(cp[:2], np.arctan2(fwd[1], fwd[0]), self.cam.hfov)
        for o in obs:
            if o.id in new:
                xy, _ = self.xy_yaw()
                tr = self.truth.get(o.id)
                err = None if tr is None else round(1000 * float(np.linalg.norm(o.p - tr["p"])), 1)
                self.first_detection[o.id] = dict(t=float(self.d.time), robot_xy=xy.round(3).tolist(),
                                                  dist_m=round(o.dist, 2), state=self.state, pos_err_mm=err)
                self.event("detected", id=o.id, dist_m=round(o.dist, 2), during=self.state, est_err_mm=err)
        if self.video is not None:
            small = self.cv2.resize(draw_detections(img, obs), (320, 180), interpolation=self.cv2.INTER_AREA)
            self.video.inset = small
        return obs, new

    def _video_markers(self, path=None):
        if self.video is None:
            return
        mk = [] if path is None else [(np.array([*p, 0.02]), [0.1, 0.5, 1.0, 1]) for p in path[::4]]
        for t in self.reg.tracks.values():
            col = [0.2, 1.0, 0.2, 1] if t.status == "visited" else [1.0, 0.85, 0.1, 1]
            mk.append((t.p + 0.06 * t.n, col))
        self.video.markers = mk

    def step(self, cmd, label):
        """One physics step with the RL policy (dt RL_DT), logging, safety checks, video."""
        m, d = self.plant, self.d
        self.pol.torques(m, d, cmd)
        mujoco.mj_step(m, d)
        hits = world_collisions(m, d, self.foot_geoms)
        for c in d.contact[:d.ncon]:
            if c.geom1 in self.stairs_geoms or c.geom2 in self.stairs_geoms:
                self.stairs_contact = True
        xy, yaw = self.xy_yaw()
        R = d.xmat[self.ix.base].reshape(3, 3)
        self.log.append(dict(t=d.time, mode=label, state=self.state, x=xy[0], y=xy[1], yaw=yaw, cmd=np.array(cmd),
                             z=d.qpos[2], tilt=float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
                             v=(R.T @ d.qvel[0:3]).copy(), clearance=self.grid.clearance_at(xy), hits=sorted(hits)))
        if d.qpos[2] < 0.12 or R[2, 2] < 0.5:
            raise RuntimeError(f"robot fell ({label}) at t={d.time:.2f}s")
        if self.video is not None and self.video.due(d.time):
            self.video.frame(d, f"ARUCO MISSION  [{self.state}: {label}]  t={d.time:5.1f}s\n{self.status_line()}")

    # ------------------------------------------------------------------ behaviours (RL policy)
    def walk(self, path, goal_yaw, label, on_frame=None, pos_tol=0.05, yaw_tol_deg=4.0, t_max=60.0, stand=0.5):
        """Follow a planned path with the RL policy; camera frames every CAPTURE_DT. on_frame(obs, new, follower)
        may return "abort". Returns dict(status = docked | aborted | timeout, ...)."""
        self.plant.opt.timestep = RL_DT
        fol = PathFollower(path, goal_yaw)
        t0, t_near, t_done, status = self.d.time, None, None, "timeout"
        self._video_markers(path)
        while self.d.time - t0 < t_max:
            xy, yaw = self.xy_yaw()
            if t_done is None:
                cmd = fol.command(xy, yaw)
                near = np.linalg.norm(fol.goal - xy) < pos_tol and abs(wrap(fol.goal_yaw - yaw)) < np.radians(yaw_tol_deg)
                t_near = (t_near or self.d.time) if near else None
                if fol.done or (t_near is not None and self.d.time - t_near > 1.0):
                    t_done, status = self.d.time, "docked"
            else:
                cmd = np.zeros(3)
                if self.d.time - t_done > stand:
                    break
            if self.d.time - self.last_capture >= CAPTURE_DT - 1e-9:
                obs, new = self.capture()
                if on_frame is not None and t_done is None and on_frame(obs, new, fol) == "abort":
                    status = "aborted"
                    break
            self.step(cmd, label)
        xy, yaw = self.xy_yaw()
        return dict(status=status, time_s=round(self.d.time - t0, 2),
                    pos_err_mm=round(1000 * float(np.linalg.norm(fol.goal - xy)), 1),
                    yaw_err_deg=round(float(np.degrees(abs(wrap(fol.goal_yaw - yaw)))), 2))

    def look_around(self, label, wz=0.8):
        """OBSERVE: a full turn in place with the RL policy, camera frames every OBSERVE_DT, then stand still."""
        self.state = "observe"
        self.plant.opt.timestep = RL_DT
        _, yaw0 = self.xy_yaw()
        turned, last, t0, n0 = 0.0, yaw0, self.d.time, len(self.reg.order_detected)
        self.last_capture = -1.0
        self._video_markers()
        while turned < 2 * np.pi and self.d.time - t0 < 25.0:
            if self.d.time - self.last_capture >= OBSERVE_DT - 1e-9:
                self.capture()
            self.step(np.array([0.0, 0.0, wz]), label)
            _, yaw = self.xy_yaw()
            turned += wrap(yaw - last)
            last = yaw
        t1 = self.d.time
        while self.d.time - t1 < 0.6:          # stop turning and settle
            self.step(np.zeros(3), label)
        self.event("observed", where=label, turned_deg=round(float(np.degrees(turned))), time_s=round(t1 - t0, 1),
                   new_markers=[i for i in self.reg.order_detected[n0:]], coverage=round(self.explorer.coverage(), 3))

    # ------------------------------------------------------------------ behaviours (stance WBC)
    def stance_confirm(self, tr, pose, standoff):
        """CONFIRM: RL -> stance WBC hand-off, hold rigid, re-measure the marker, WBC -> RL hand-back."""
        self.state = "confirm"
        m, d, ix = self.plant, self.d, self.ix
        m.opt.timestep = WBC_DT
        hold = StanceHold(self.cfg, d.qpos.copy(), d.time)
        wbc = StanceWBC(self.ctrl, self.cfg, **self.cfg.wbc_gains)
        obs_t, base_v, base_p, last = [], [], [], -1.0
        while not hold.done(d.time):
            ph = hold.phase(d.time)
            if ph == "hold" and d.time - last >= 0.2 - 1e-9:
                last = d.time
                img, cp, cR = self.cam.capture(m, d)
                obs = self.det.detect(img, cp, cR, t=d.time)
                self.reg.update(obs)
                obs_t += [o for o in obs if o.id == tr.id]
                if self.video is not None:
                    self.video.inset = self.cv2.resize(draw_detections(img, obs), (320, 180),
                                                       interpolation=self.cv2.INTER_AREA)
            tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), hold.ref(d.time))
            d.ctrl[:] = tau[ix.act_dof]
            mujoco.mj_step(m, d)
            if ph == "hold":
                base_v.append(np.linalg.norm(d.qvel[0:3]))
                base_p.append(d.qpos[0:3].copy())
            if d.qpos[2] < 0.12:
                raise RuntimeError("robot fell during the stance hold")
            if self.video is not None and self.video.due(d.time):
                self.video.frame(d, f"ARUCO MISSION  [confirm at marker {tr.id}: stance WBC {ph}]  t={d.time:5.1f}s\n"
                                    f"{self.status_line()}")
        m.opt.timestep = RL_DT
        self.pol.reset()
        xy, yaw = self.xy_yaw()
        out = dict(seen=len(obs_t), wbc_qp_failures=wbc.fail,
                   base_speed_rms_mm_s=round(1000 * float(np.sqrt(np.mean(np.square(base_v)))), 2),
                   base_drift_mm=round(1000 * float(np.max(np.linalg.norm(np.array(base_p) - base_p[0], axis=1))), 2))
        if not obs_t:
            return dict(out, confirmed=False, reason="marker not seen from the target pose")

        def align(p, n):
            R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            rel = R @ (p[:2] - xy)
            nh = n[:2] / np.linalg.norm(n[:2])
            return dict(distance_m=round(float(rel[0]), 3), lateral_m=round(float(rel[1]), 3),
                        heading_err_deg=round(float(np.degrees(wrap(yaw - np.arctan2(-nh[1], -nh[0])))), 2))
        belief = align(tr.p, tr.n)
        gt = self.truth[tr.id]
        truth = align(gt["p"], gt["n"])
        ok = (abs(belief["lateral_m"]) < ALIGN_TOL["lateral_m"]
              and abs(belief["distance_m"] - standoff) < ALIGN_TOL["distance_m"]
              and abs(belief["heading_err_deg"]) < ALIGN_TOL["heading_deg"]
              and out["base_speed_rms_mm_s"] < ALIGN_TOL["base_speed_mm_s"] and wbc.fail == 0)
        return dict(out, confirmed=bool(ok), standoff_m=standoff, alignment_belief=belief, alignment_truth=truth)

    # ------------------------------------------------------------------ high level
    def go_to_marker(self, tr, max_replans=6):
        """NAVIGATE to the best approach pose of a marker; replan when the marker estimate moves it."""
        self.state = "navigate"
        if tr.status == "visited":
            self.revisits += 1
        res, plans = None, []
        for k in range(max_replans + 1):
            xy, _ = self.xy_yaw()
            c0 = time.perf_counter()
            path, info, pose, standoff, off = plan_to_marker(self.grid, xy, tr.p, tr.n, standoffs=(self.args.standoff,
                                                                                                  self.args.standoff + 0.15))
            plans.append(dict(plan_ms=round(1000 * (time.perf_counter() - c0), 1), length_m=round(info["length"], 2),
                              min_clearance_m=round(info["min_clearance"], 3), standoff_m=standoff, yaw_offset_deg=off))
            self.paths.append((f"to {tr.id}", path))
            self.event("plan", target=tr.id, attempt=k, length_m=round(info["length"], 2), pose=pose.round(2).tolist())

            def on_frame(obs, new, fol, pose=pose, standoff=standoff, off=off):
                if not any(o.id == tr.id for o in obs):
                    return None
                p2 = [q for q, d_, o_ in approach_poses(tr.p, tr.n, standoffs=(standoff,), yaw_offsets_deg=(off,))][0]
                far = np.linalg.norm(fol.goal - self.xy_yaw()[0]) > 0.5
                moved = np.linalg.norm(p2[:2] - pose[:2]) > 0.10 or abs(wrap(p2[2] - pose[2])) > np.radians(6)
                return "abort" if (far and moved) else None
            res = self.walk(path, pose[2], f"to marker {tr.id}", on_frame=on_frame if k < max_replans else None)
            if res["status"] == "docked":
                return res, pose, standoff, plans
            if res["status"] == "aborted":
                self.event("replan", target=tr.id, reason="marker estimate refined")
                continue
            break
        return res, None, None, plans

    def verify(self, tr, look_s=1.5):
        """VERIFY a candidate (seen, not yet confirmed): walk to a viewpoint ~1.1 m in front of it, stand and look.
        It becomes a target if real (3 consistent views on a map surface), otherwise it is rejected."""
        self.state = "verify"
        xy, _ = self.xy_yaw()
        self.event("verify", id=tr.id, views=tr.n_obs, estimate=tr.p.round(2).tolist())
        try:
            path, info, pose, _, _ = plan_to_marker(self.grid, xy, tr.p, tr.n, standoffs=(1.1, 1.4, 0.9))
        except RuntimeError as e:
            self.reg.reject(tr.id)
            self.event("verify", id=tr.id, result=f"rejected (no viewpoint: {e})")
            return
        self.paths.append((f"verify {tr.id}", path))

        def on_frame(obs, new, fol):
            return "abort" if tr.status == "detected" else None
        self.walk(path, pose[2], f"verify {tr.id}", on_frame=on_frame, pos_tol=0.10, yaw_tol_deg=8.0)
        t0 = self.d.time
        while tr.status == "candidate" and self.d.time - t0 < look_s:      # stand still and look
            if self.d.time - self.last_capture >= OBSERVE_DT - 1e-9:
                self.capture()
            self.step(np.zeros(3), f"verify {tr.id}")
        if tr.status == "candidate":
            self.reg.reject(tr.id)
        self.event("verify", id=tr.id, result="confirmed" if tr.status == "detected" else "rejected (false positive)")

    def explore(self):
        """EXPLORE: walk to the best viewpoint (abort on a new marker), then OBSERVE. False if nothing is left."""
        self.state = "explore"
        xy, _ = self.xy_yaw()
        c0 = time.perf_counter()
        vp = self.explorer.next_viewpoint(xy)
        ms = 1000 * (time.perf_counter() - c0)
        if vp is None:
            self.event("explore", result="no viewpoint with unseen curved surface left")
            return False
        path, info = plan_path(self.grid, xy, vp["xy"])
        goal_yaw = float(np.arctan2(*(path[-1] - path[max(0, len(path) - 4)])[::-1])) if len(path) > 1 else 0.0
        self.viewpoints.append(vp["xy"])
        self.paths.append(("explore", path))
        self.event("explore", viewpoint=vp["xy"].round(2).tolist(), gain=round(vp["gain"], 1),
                   dist_m=round(vp["dist"], 2), select_ms=round(ms))

        def on_frame(obs, new, fol):
            return "abort" if any(self.reg.tracks[i].status == "detected" for i in new) else None
        res = self.walk(path, goal_yaw, "exploring", on_frame=on_frame, pos_tol=0.15, yaw_tol_deg=15.0)
        if res["status"] == "aborted":
            self.event("explore", result="new marker detected on the way")
            return True
        self.look_around("at viewpoint")
        return True

    def run(self):
        a, t_wall = self.args, time.perf_counter()
        self.event("start", robot=a.robot, loco=self.loco, markers_expected=self.reg.n_expected)
        self.look_around("start")                                     # step 1: initial observation
        visits, explorations, failures = [], 0, []
        while not self.reg.complete and self.d.time < a.max_time:
            tr = self.reg.next_target(self.xy_yaw()[0])
            if tr is None:
                cand = self.reg.next_candidate(self.xy_yaw()[0])
                if cand is not None:                                  # half-seen marker: take a closer look
                    self.verifications += 1
                    self.verify(cand)
                    continue
                explorations += 1                                     # nothing known: explore
                if not self.explore():
                    break
                continue
            self.event("target", id=tr.id, estimate=tr.p.round(3).tolist(), views=tr.n_obs)
            v = dict(id=tr.id, surface=self.truth[tr.id]["surface"], first_detection=self.first_detection.get(tr.id))
            confirmed = False
            for attempt in range(3):                                  # step 2: navigate + align + confirm
                try:
                    res, pose, standoff, plans = self.go_to_marker(tr)
                except RuntimeError as e:
                    v["error"] = str(e)
                    break
                v.setdefault("navigation", []).append(dict(walk=res, plans=plans))
                if res["status"] != "docked":
                    continue
                conf = self.stance_confirm(tr, pose, standoff)
                v.setdefault("confirm", []).append(conf)
                self.event("confirm", id=tr.id, attempt=attempt, confirmed=conf["confirmed"],
                           alignment=conf.get("alignment_belief"))
                if conf["confirmed"]:
                    confirmed = True
                    break
            gt = self.truth[tr.id]
            v.update(views=tr.n_obs, estimate_err_mm=round(1000 * float(np.linalg.norm(tr.p - gt["p"])), 1),
                     normal_err_deg=round(float(np.degrees(np.arccos(np.clip(tr.n @ gt["n"], -1, 1)))), 2))
            visits.append(v)
            if not confirmed:
                self.reg.mark_unreachable(tr.id, v.get("error", "alignment not confirmed"))
                failures.append(tr.id)
                self.event("unreachable", id=tr.id)
                continue
            self.reg.mark_visited(tr.id, float(self.d.time), dict(alignment=v["confirm"][-1]["alignment_truth"]))
            self.event("visited", id=tr.id, n_visited=self.reg.n_visited)
            if not self.reg.complete:
                self.look_around(f"marker {tr.id}")                  # step 3: re-observe at the target
        self.state = "done"
        complete = self.reg.complete
        banner = (f"TASK COMPLETE: all {self.reg.n_expected} ArUco markers visited {self.reg.order_visited}" if complete
                  else f"TASK INCOMPLETE: visited {self.reg.order_visited} of {self.reg.n_expected}")
        self.event("done", complete=complete, sim_time_s=round(float(self.d.time), 1))
        print("\n" + "=" * len(banner) + "\n" + banner + "\n" + "=" * len(banner) + "\n", flush=True)
        if self.video is not None:
            for _ in range(int(2 * FPS)):
                self.video.frame(self.d, f"ARUCO MISSION  {banner}\n{self.status_line()}")
        return self.summary(visits, explorations, failures, complete, banner, time.perf_counter() - t_wall)

    # ------------------------------------------------------------------ results
    def summary(self, visits, explorations, failures, complete, banner, wall_s):
        moving = [r for r in self.log if np.linalg.norm(r["cmd"][:2]) > 0.2]
        visited_order = self.reg.order_visited
        surfaces = {}
        for mk in self.world.markers:
            s = self.world.surface(mk.surface)
            surfaces[mk.id] = ("vertical cylinder" if isinstance(s, Cylinder) else "horizontal pipe (cylinder)"
                               if isinstance(s, Pipe) else "sphere" if isinstance(s, Ellipsoid) and len(set(s.radii)) == 1
                               else "ellipsoid")
        observed_at = [e["where"] for e in self.events if e["event"] == "observed"]
        conf = [v["confirm"][-1] for v in visits if v.get("confirm")]
        checks = {
            "1_four_markers_spawned": len({mk.id for mk in self.world.markers}) == 4
            and sum((mujoco.mj_id2name(self.plant, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("aruco_")
                    for i in range(self.plant.ngeom)) == 4,
            "2_one_marker_per_curved_surface": len({mk.surface for mk in self.world.markers}) == len(self.world.markers),
            "3_detected_and_pose_estimated": all(v.get("estimate_err_mm", 1e9) < 50 for v in visits) and len(visits) > 0,
            "4_navigated_on_map": all(any(n["walk"]["status"] == "docked" for n in v.get("navigation", []))
                                      for v in visits if v["id"] in visited_order),
            "5_no_obstacle_or_stairs_contact": not self.stairs_contact
            and not any(r["hits"] for r in self.log),
            "6_stabilised_and_aligned": len(conf) > 0 and all(c["confirmed"] for c in conf),
            "7_reobserved_at_targets": all(f"marker {i}" in observed_at for i in visited_order[:-1]),
            "8_explored_when_nothing_visible": explorations > 0,
            "9_no_revisits": len(set(visited_order)) == len(visited_order) and self.revisits == 0,
            "10_terminated_after_all_four": complete and set(visited_order) == set(self.truth),
        }
        return dict(
            banner=banner, complete=complete, robot=self.args.robot, loco=self.loco, world=self.args.world,
            seed=self.args.seed, sim_time_s=round(float(self.d.time), 1), wall_time_s=round(wall_s, 1),
            order_detected=self.reg.order_detected, order_visited=visited_order, unreachable=failures,
            unconfirmed_candidates=self.reg.candidates(), verifications=self.verifications,
            rejected_false_positives=[i for i, t in self.reg.tracks.items() if t.status == "rejected"],
            markers={mid: dict(surface=t["surface"], surface_type=surfaces[mid], p=np.round(t["p"], 3).tolist(),
                               n=np.round(t["n"], 3).tolist()) for mid, t in self.truth.items()},
            visits=visits, revisits=self.revisits, explorations=explorations, viewpoints=[np.round(v, 2).tolist() for v in self.viewpoints],
            surface_coverage=round(self.explorer.coverage(), 3),
            walking=dict(min_clearance_m=round(float(min(r["clearance"] for r in self.log)), 3),
                         world_contacts=sorted({h for r in self.log for h in r["hits"]}),
                         stairs_contact=self.stairs_contact,
                         tilt_max_deg=round(float(max(r["tilt"] for r in self.log)), 1),
                         speed_tracking_rms=round(float(np.sqrt(np.mean([np.sum((r["cmd"][:2] - r["v"][:2]) ** 2)
                                                                         for r in moving]))), 3) if moving else None),
            checks=checks, events=self.events)

    def plot_map(self, path_png, summary):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        w, g = self.world, self.grid
        fig, ax = plt.subplots(figsize=(11, 8.4))
        ax.imshow(np.where(g.occ, 2, np.where(g.free, 0, 1)).T, origin="lower", extent=list(w.bounds), cmap="Greys",
                  vmax=2.6)
        for st in w.stairs:
            c, a = st.center_xy(), np.array([np.cos(st.yaw), np.sin(st.yaw)])
            b = np.array([-a[1], a[0]])
            q = [c + sx * st.length / 2 * a + sy * st.width / 2 * b for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ax.add_patch(Polygon(q, closed=True, color="#d4a017", alpha=0.9, label="stairs (not traversable)"))
        sc = self.explorer
        ax.scatter(sc.xy[sc.w >= 1, 0], sc.xy[sc.w >= 1, 1], s=3, c=np.where(sc.seen[sc.w >= 1], "#2da44e", "#cf222e"),
                   label="curved surface seen / unseen")
        cols = {"observe": "#8250df", "navigate": "#1f6feb", "explore": "#bf8700", "verify": "#1a7f37", "confirm": "#cf222e"}
        tr = np.array([(r["x"], r["y"]) for r in self.log])
        st = np.array([r["state"] for r in self.log])
        for k, col in cols.items():
            msk = st == k
            if msk.any():
                ax.scatter(tr[msk, 0], tr[msk, 1], s=1, c=col, label=f"robot ({k})")
        for mid, t in self.truth.items():
            p, n = t["p"], t["n"]
            ax.annotate("", p[:2] + 0.45 * n[:2], p[:2], arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
            k = self.reg.order_visited.index(mid) + 1 if mid in self.reg.order_visited else "-"
            ax.annotate(f" id {mid} (#{k})", p[:2], fontsize=9, fontweight="bold")
            if mid in self.reg.tracks:
                ax.plot(*self.reg.tracks[mid].p[:2], "x", color="#2da44e", ms=9, mew=2)
        for v in self.viewpoints:
            ax.plot(*v, "*", color="#bf8700", ms=14, mec="k")
        ax.plot(*w.start[:2], "ko", ms=7)
        ax.set_title(f"ArUco mission ({summary['robot']}, {summary['loco']}): {summary['banner']}", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        h, lab = ax.get_legend_handles_labels()
        u = dict(zip(lab, h))
        ax.legend(u.values(), u.keys(), fontsize=7, loc="lower left", markerscale=4)
        fig.tight_layout()
        fig.savefig(path_png, dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="go2_z1")
    ap.add_argument("--loco", default="auto")
    ap.add_argument("--world", choices=["aruco_workshop"], default="aruco_workshop")
    ap.add_argument("--seed", type=int, default=0, help="camera pixel-noise seed")
    ap.add_argument("--marker-size", type=float, default=0.12)
    ap.add_argument("--pixel-noise", type=float, default=2.0, help="std of the Gaussian image noise [0-255]")
    ap.add_argument("--standoff", type=float, default=0.75, help="base centre -> marker distance at a target [m]")
    ap.add_argument("--range", type=float, default=3.5, help="range at which a surface counts as checked [m]")
    ap.add_argument("--max-time", type=float, default=400.0, help="sim time limit [s]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-speed", type=float, default=2.0, help="video playback speed-up")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    mission = ArucoMission(args)
    summary = mission.run()
    out = os.path.join(ROOT, "results", "aruco")
    os.makedirs(out, exist_ok=True)
    name = f"{args.robot}{args.tag}"
    with open(os.path.join(out, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    mission.plot_map(os.path.join(out, f"map_{name}.png"), summary)
    print(json.dumps(summary["checks"], indent=1))
    if mission.video is not None and mission.video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission_{name}.mp4"), mission.video.frames, fps=FPS, quality=6,
                        macro_block_size=1)
    sys.exit(0 if summary["complete"] else 1)


if __name__ == "__main__":
    main()
