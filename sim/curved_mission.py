"""ArUco work-order mission on non-flat surfaces (sim/curved_world.py): find the four markers, do each one's work order.

One flow, five layers - each answers one question and trusts the layers below it:

    1. LOCALISE     "where am I?"        always on, while walking and standing: leg/IMU odometry (drifts) corrected by
                                         depth-scan matching to the known map (sim/perception/map_localiser.py); the
                                         stairs riser fixes the position along the corridor. Markers are measured, not
                                         used to localise (their first, far-away estimates are less accurate).
                                         Paused only while docking (the work is placed relative to the marker there).
    2. WORLD MODEL  "what do I know?"    the map + a marker registry: seen -> confirmed (2 consistent views within 4 m)
                                         -> worked | failed; per marker its measured surface and reach plan; which map
                                         surfaces the camera has looked at (sim/planning/explorer.py).
    3. DECIDE       "what next?"         --strategy work_as_found (default): a confirmed, unworked marker? -> work the
                                         cheapest one (A* cost, stairs = portal edge); else -> explore; nothing left ->
                                         done. --strategy map_first: explore until all markers are found, then work
                                         them in the order of the cheapest tour.
                                         EXPLORE = look around here (one 360 deg turn) if this spot was not looked at,
                                         else walk to the viewpoint showing the most unseen surface; a walk stops as soon
                                         as a new marker is confirmed (work_as_found).
    4. EXECUTE      "do the job"         the same six steps for every marker, whatever the surface:
                                           1 approach    walk to the survey pose; re-aim when the marker estimate moves
                                           2 survey      measure the marker from here (re-acquire if not in view), face
                                                         the area, depth frames -> surface model w(u, v)
                                           3 plan reach  IK over the measured surface -> standoff + patches
                                           4 dock        per patch: measure the marker from a pre-dock pose, dock on it,
                                                         settle, correct until within 2.5 cm / 2 deg
                                           5 work        hand-off to the stance WBC, 5 N arm admittance trace, hand back
                                           6 check       contact held? (robot's own force sensor) -> one redo; what was
                                                         out of reach from the actual stance -> a new patch
    5. CONTROL      "move"               walking: blind MoE-CTS RL policy (climbs the stairs); standing work: stance WBC +
                                         arm admittance (sim/aruco_mission_demo.py, sim/area_task.py, unchanged)

    python sim/curved_mission.py                                  # -> results/curved_mission/
    python sim/curved_mission.py --no-video --seed 2 --tag _seed2
    python sim/curved_mission.py --strategy map_first --tag _mapfirst
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curved_world as CW  # noqa: E402
from aruco_mission_demo import (DETECT_EVERY, MAX_REMAINDERS, PRE_DOCK, Mission, PlanError, Task,  # noqa: E402
                                plot_map, plot_paint)
from area_task import SurfacePatch, plan_stance, reachable_span, stance_pose  # noqa: E402
from perception.aruco_detect import FOVY, H as CAM_H, MOUNT as CAM_MOUNT, W as CAM_W  # noqa: E402
from perception.map_localiser import MapLocaliser  # noqa: E402
from perception.odometry import wrap  # noqa: E402
from perception.work_area import fit_area_surface  # noqa: E402
from planning.explorer import Explorer  # noqa: E402
from planning.task_planner import STAIRS_PENALTY  # noqa: E402
from wbc.robots import ROOT  # noqa: E402

LOC_DT = 0.5               # [s] map-matching period
DOCK_TOL = (0.025, 2.0)    # [m, deg] settled dock accepted
LOOKED_RADIUS = 1.5        # [m] a spot counts as looked around within this distance of an earlier look-around


class PortalExplorer(Explorer):
    """Explorer whose reachability (BFS over free cells) also crosses the staircase portal: the stairs are blocked in
    the grid, so the far side is reached from the portal's far end at the cost of the crossing."""

    def __init__(self, planner, world, **kw):
        self.planner = planner
        super().__init__(planner.grid, world, **kw)

    def _reach(self, start_xy):
        D = super()._reach(start_xy)
        a, b = self.planner.portal
        cross = float(np.linalg.norm(b - a)) + STAIRS_PENALTY
        for p, q in ((a, b), (b, a)):
            dp, dq = D[self.grid.to_cell(p)], D[self.grid.to_cell(q)]
            if np.isfinite(dp) and not np.isfinite(dq):
                D = np.minimum(D, super()._reach(q) + dp + cross)
        return D

    def mark_looked(self, xy):
        """Everything visible from xy counts as looked at (after a look-around there, or if xy cannot be reached)."""
        idx = np.flatnonzero(~self.seen)
        self.seen[idx[self._visible(np.asarray(xy, float), idx)]] = True


class CurvedMission(Mission):
    work_orders = CW.WORK_ORDERS
    marker_size = CW.MARKER_SIZE

    def make_world(self, args):
        return CW.curved_world(decimate=args.terrain_decimate, panel_seed=args.panel_seed)

    def __init__(self, args):
        args.explore = True        # v8 walk(): stops when the queue head changes (used for "new marker while exploring")
        super().__init__(args)
        # 1 LOCALISE
        solid, exclude = self.world.localiser_map()
        self.localiser = MapLocaliser(self.world, solid=solid, exclude=exclude)
        self.t_loc = -1.0
        # 2 WORLD MODEL
        self.rgb.max_dist = args.max_marker_dist   # bent stickers: 1-5 cm within 4 m, worse beyond
        # a surface counts as looked at only within 25 deg of its normal (in the map's 2-D view): the bent stickers are
        # read up to ~30 deg off their normal (none at 45 deg), and a sticker below a tank's equator also faces 33 deg
        # down, which the 2-D map does not see - with 35 deg the tank's marker side was counted as looked at too early
        self.explorer = PortalExplorer(self.planner, self.world, range_m=args.range, min_cos=np.cos(np.radians(25)))
        self.hfov = 2 * np.arctan(CAM_W / CAM_H * np.tan(np.radians(FOVY) / 2))
        self.status = {}           # marker id -> "confirmed" | "worked" | "failed"
        self.attempts, self.marker_results, self.order_worked = {}, {}, []
        self.looked_at, self.viewpoints = [], []
        # 3 DECIDE
        self.mode, self.current = "start", None
        self.decisions = []
        self.n_observe = self.n_explore = 0
        # live 3-D window (--viewer; macOS: run with mjpython). Synced ~30x per wall-clock second from the two places
        # the simulation steps: walking/standing (step_rl) and the stance-WBC trace loop (frame).
        self.viewer, self._t_sync = None, 0.0
        if getattr(args, "viewer", False):
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m, self.d)

    def _sync_viewer(self):
        if getattr(self, "viewer", None) is None or time.perf_counter() - self._t_sync < 1 / 30:
            return
        if not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        self._t_sync = time.perf_counter()
        b = self.d.qpos[0:3]
        self.viewer.cam.lookat[:] = b                 # follow the robot
        self.viewer.sync()

    def frame(self, *a, **kw):
        self._sync_viewer()
        return super().frame(*a, **kw)

    # ================================================================== 1 LOCALISE
    def step_rl(self, cmd):
        R = super().step_rl(cmd)
        self._sync_viewer()
        if self.reloc_on and self.d.time - self.t_loc >= LOC_DT and self.pol.k % 10 == 0:
            self.t_loc = self.d.time
            pts = self.depth.capture(self.m, self.d)[0]
            if len(pts):
                self.localiser.correct(self.odo, self.odo.to_est(pts))
        return R

    # ================================================================== 2 WORLD MODEL
    def detect(self, interrupt_ok=False):
        # v8: ArUco + PnP -> marker registry. v8 also re-localised on markers (pulling the pose onto each marker's first,
        # often far-away estimate); map matching is more accurate, so here markers are measured, not used to localise
        on, self.reloc_on = self.reloc_on, False
        try:
            new = super().detect(interrupt_ok)
        finally:
            self.reloc_on = on
        xy, yaw = self.pose()
        self.explorer.mark_seen(xy + CAM_MOUNT[0] * np.array([np.cos(yaw), np.sin(yaw)]), yaw, self.hfov)
        return new

    def _add_survey(self, mid):
        """v8 hook, called when a marker is confirmed: record it (the decide layer picks what to do with it)."""
        self.status[mid] = "confirmed"
        p, _ = self.markers.pose(mid)
        self.planner.add_obstacle(("circle", p[0], p[1], 0.12))      # the marker is on *something*
        if self.mode == "explore" and self.args.strategy == "work_as_found":
            self.queue = [str(mid)]                                  # stops the exploration walk (v8 walk())

    def reorder(self):
        return self.queue

    def unworked(self):
        return [m for m, s in self.status.items() if s == "confirmed"]

    def survey_pose(self, mid):
        f = self.frame_est(mid)
        c = f.o + f.W / 2 * f.ex
        return np.array([*self.free_near(c[:2] + self.survey_distance(f) * f.ez[:2]), np.arctan2(-f.ez[1], -f.ez[0])])

    # ================================================================== 3 DECIDE
    def decide(self):
        """-> ("work", mid) | ("look", None) | ("explore", viewpoint) | ("done", reason)"""
        xy, _ = self.pose()
        todo = self.unworked()
        all_found = len(self.status) == len(self.work_orders)
        if todo and (self.args.strategy == "work_as_found" or all_found):
            if self.args.strategy == "map_first" and len(todo) > 1:
                keys, _ = self.planner.order(xy, [(str(m), self.survey_pose(m), m) for m in todo])
                return "work", int(keys[0]) if keys else todo[0]
            return "work", min(todo, key=lambda m: self.planner.cost(xy, self.survey_pose(m)[:2]))
        retry = [m for m, s in self.status.items() if s == "failed" and self.attempts.get(m, 0) < 2]
        if all_found and retry:
            return "work", retry[0]
        if all_found:
            return "done", "all markers handled"
        if not any(np.linalg.norm(xy - p) < LOOKED_RADIUS for p in self.looked_at):
            return "look", None
        vp = self.explorer.next_viewpoint(xy)
        if vp is not None:
            return "explore", vp
        if retry:
            return "work", retry[0]
        if todo:
            return "work", todo[0]
        return "done", "no viewpoint with unseen surface left"

    def run(self):
        a = self.args
        self.log(f"start: work orders {sorted(self.work_orders)}, strategy {a.strategy}; map known, markers unknown")
        while self.d.time < a.max_time:
            action, arg = self.decide()
            self.decisions.append((round(self.d.time, 1), action, arg if not isinstance(arg, dict) else
                                   np.round(arg["xy"], 2).tolist()))
            if action == "done":
                self.log(f"[decide] done: {arg}")
                break
            if action == "work":
                self.log(f"[decide] work marker {arg}")
                self.execute(arg)
            elif action == "look":
                self.log("[decide] nothing to work on, this spot not looked at -> look around")
                self.look_around_here()
            else:
                self.log(f"[decide] explore -> viewpoint {np.round(arg['xy'], 2)} (unseen surface {arg['gain']:.0f}, "
                         f"{arg['dist']:.1f} m)")
                self.explore(arg)
        done = [m for m, s in self.status.items() if s == "worked"]
        if len(done) == len(self.work_orders):
            self.banner = f"TASK COMPLETE: all {len(done)} markers worked, order {self.order_worked}"
        elif len(self.status) == len(self.work_orders):
            self.banner = (f"TASK FINISHED: {len(done)}/{len(self.work_orders)} worked, failed "
                           f"{[m for m, s in self.status.items() if s == 'failed']}")
        else:
            self.banner = f"TASK INCOMPLETE: worked {done}, found {sorted(self.status)}"
        self.log(self.banner)
        self.stand(1.5)

    # ------------------------------------------------------------------ explore behaviours
    def look_around_here(self, wz=0.7):
        """One full turn in place (heading drift is corrected by the localiser, no back-and-forth needed)."""
        self.mode = "explore"
        self.n_observe += 1
        n0 = len(self.status)
        _, last = self.pose()
        turned, t0 = 0.0, self.d.time
        cmd = np.array([0.0, 0.0, wz])
        while abs(turned) < 2 * np.pi and self.d.time - t0 < 2 * np.pi / wz * 1.6 + 2.0:
            self.step_rl(cmd)
            if self.d.time - self.t_detect >= DETECT_EVERY / 2:
                self.t_detect = self.d.time
                self.detect()
            _, yaw = self.pose()
            turned += wrap(yaw - last)
            last = yaw
            self.frame("look around", cmd)
        self.stand(0.5, detect=True)
        xy, _ = self.pose()
        self.looked_at.append(xy)
        self.explorer.mark_looked(xy)
        e = self.odo.error()
        self.log(f"look around: {np.degrees(turned):.0f} deg, new markers {sorted(set(self.status) - set(list(self.status)[:n0]))}, "
                 f"surface looked at {100 * self.explorer.coverage():.0f} %, localisation error {100 * e[0]:.0f} cm / "
                 f"{e[1]:.1f} deg")

    def explore(self, vp):
        self.mode = "explore"
        self.n_explore += 1
        self.queue = []
        xy, _ = self.pose()
        goal = self.free_near(vp["xy"])
        self.viewpoints.append(goal)
        try:
            r = self.go_pose(np.array([*goal, float(np.arctan2(*(goal - xy)[::-1]))]), "explore", approach=0.0,
                             tol=(0.25, 25.0))
        except PlanError as ex:
            self.log(f"explore: viewpoint unreachable ({ex})")
            self.explorer.mark_looked(goal)
            return
        if r == "interrupted":
            self.log("explore: new marker confirmed on the way")
            return
        self.look_around_here()

    # ================================================================== 4 EXECUTE
    def execute(self, mid):
        """The six steps for one marker. Records the result in the world model."""
        self.mode, self.current, self.queue = "work", mid, [str(mid)]
        self.attempts[mid] = self.attempts.get(mid, 0) + 1
        t0, steps = self.d.time, {}
        res = dict(status="failed", attempt=self.attempts[mid], patches={})
        try:
            ok = self.approach(mid)                                              # 1
            steps["approach_s"] = round(self.d.time - t0, 1)
            meas = self.survey(mid) if ok else None                              # 2
            steps["survey_s"] = round(self.d.time - t0 - steps["approach_s"], 1)
            plan = self.plan_reach(mid, *meas) if meas else None                 # 3
            if plan is not None:
                f, surf = meas
                patches = [Task(f"{mid}.{i}", mid, "patch", stance_pose(f, plan.w_ref, plan.D, s), mid,
                                info=dict(surf=surf, plan=plan, u0=a, u1=b, s=s, left=MAX_REMAINDERS, redo=1))
                           for i, (a, b, s) in enumerate(plan.patches)]
                while patches:
                    task = patches.pop(0)
                    try:
                        docked = self.dock(task)                                 # 4
                    except PlanError as ex:
                        self.log(f"[4 dock] {task.key}: no path ({ex})")
                        docked = None
                    r = self.work(task, docked) if docked else "failed"          # 5
                    patches = self.check(task, docked, r) + patches              # 6
                    res["patches"][task.key] = r
                ok_patches = [k for k, r in res["patches"].items() if r == "done"]
                res["status"] = "done" if ok_patches else "failed"
        finally:
            self.reloc_on = True
            self.current = None
        res.update(steps, time_s=round(self.d.time - t0, 1))
        self.marker_results[mid] = res
        self.status[mid] = "worked" if res["status"] == "done" else "failed"
        (self.done if res["status"] == "done" else self.failed).append(str(mid))
        if mid not in self.order_worked:
            self.order_worked.append(mid)
        self.log(f"[execute] marker {mid}: {res['status'].upper()} in {res['time_s']:.0f} s "
                 f"({sum(s == 'worked' for s in self.status.values())}/{len(self.work_orders)} worked)")
        self.step_back()

    def step_back(self, dist=0.6):
        """Leave the work stance before the next walk / turn (the arm and head must not sweep the workpiece)."""
        xy, yaw = self.pose()
        self.line_to(np.array([*self.free_near(xy - dist * np.array([np.cos(yaw), np.sin(yaw)])), yaw]), "step back",
                     tol=(0.08, 6.0))

    # ---- 1 approach
    def approach(self, mid):
        """Walk to the survey pose; closer sightings on the way refine the marker estimate, so on arrival the pose is
        recomputed and walked again if it moved > 15 cm / 8 deg (at most twice)."""
        pose = self.survey_pose(mid)
        try:
            for k in range(3):
                self.go_pose(pose, f"[1 approach] marker {mid}", approach=0.6)
                new = self.survey_pose(mid)
                moved, turned = np.linalg.norm(new[:2] - pose[:2]), abs(np.degrees(wrap(new[2] - pose[2])))
                if moved < 0.15 and turned < 8.0:
                    return True
                self.log(f"[1 approach] marker {mid}: estimate refined on the way, pose moved {100 * moved:.0f} cm / "
                         f"{turned:.0f} deg - re-plan")
                pose = new
            return True
        except PlanError as ex:
            self.log(f"[1 approach] marker {mid}: no path ({ex})")
            return False

    # ---- 2 survey
    def reacquire(self, mid):
        """Marker not in view from where the robot thinks it should be: sweep +-70 deg, then face its estimate."""
        self.log(f"[2 survey] marker {mid} not in view - re-acquire (sweep +-70 deg)")
        for wz, T in ((0.7, 1.75), (-0.7, 3.5), (0.7, 1.75)):
            t0 = self.d.time
            while self.d.time - t0 < T:
                self.step_rl(np.array([0.0, 0.0, wz]))
                if self.d.time - self.t_detect >= DETECT_EVERY / 2:
                    self.t_detect = self.d.time
                    self.detect()
                self.frame("re-acquire", np.array([0, 0, wz]))
        p, _ = self.markers.pose(mid)
        xy, _ = self.pose()
        self.line_to(np.array([*xy, np.arctan2(*(p[:2] - xy)[::-1])]), "face the marker", tol=(0.1, 3.0))
        return self.fresh_frame(mid, 1.2, fallback=False)

    def survey(self, mid):
        f = self.fresh_frame(mid, 1.2, fallback=False)         # the marker relative to the robot, from here
        if f is False:
            f = self.reacquire(mid)
        if f is False:
            self.log(f"[2 survey] marker {mid}: not found from the survey pose")
            return None
        pose = self.survey_pose_from(f)
        xy, yaw = self.pose()
        if np.linalg.norm(xy - pose[:2]) > 0.08 or abs(wrap(yaw - pose[2])) > np.radians(4):
            self.line_to(pose, f"[2 survey] marker {mid}: face the area")
            f = self.fresh_frame(mid, 1.0, fallback=f)
        clouds = []
        self.stand(1.2, on_tick=lambda: clouds.append(self.depth.capture(self.m, self.d)[0]) if len(clouds) < 16 else None)
        pts = self.odo.to_est(np.concatenate(clouds))
        try:
            surf = fit_area_surface(pts, f)
        except RuntimeError as ex:
            self.log(f"[2 survey] marker {mid}: {ex}")
            return None
        q = f.to_area(pts)
        self.cloud = (pts[(q[:, 0] > -0.05) & (q[:, 0] < f.W + 0.05) & (q[:, 1] > -0.05) & (q[:, 1] < f.H + 0.05)
                          & (np.abs(q[:, 2]) < 0.3)][::2], [0.2, 1.0, 0.3, 1])
        acc = self.model_accuracy(mid, f, surf)
        self.log(f"[2 survey] marker {mid}: surface model {surf.info['cells']} cells, depth "
                 f"{surf.info['depth_range_mm'][0]:.0f}..{surf.info['depth_range_mm'][1]:.0f} mm; vs real surface "
                 f"{acc['depth_rms_mm']:.1f} mm / {acc['normal_rms_deg']:.1f} deg RMS")
        w = surf.w(*np.meshgrid(np.linspace(0, f.W, 9), np.linspace(0, f.H, 3)))
        w0, w1 = float(w.min()) - 0.20, float(w.max())        # what was seen is an obstacle (+20 cm assumed body)
        c = f.to_world(f.W / 2, 0.0, (w0 + w1) / 2)
        self.planner.add_obstacle(("box", c[0], c[1], f.W / 2 + 0.02, (w1 - w0) / 2, float(np.arctan2(f.ex[1], f.ex[0]))))
        self.summary["surveys"][str(mid)] = dict(model=surf.info, accuracy=acc)
        return f, surf

    def survey_pose_from(self, f):
        c = f.o + f.W / 2 * f.ex
        return np.array([*self.free_near((c + self.survey_distance(f) * f.ez)[:2]), np.arctan2(-f.ez[1], -f.ez[0])])

    # ---- 3 plan reach
    def plan_reach(self, mid, f, surf):
        c0 = time.perf_counter()
        try:
            fz = self.floor_z()
            plan = plan_stance(self.cfg, self.ctrl, f, surf, lambda xy, yaw: self.planner.stance_ground(xy, yaw, fz))
        except RuntimeError as ex:
            self.log(f"[3 plan reach] marker {mid}: {ex}")
            return None
        self.log(f"[3 plan reach] marker {mid} ({time.perf_counter() - c0:.1f} s): standoff {plan.D:.2f} m, reachable "
                 f"{1000 * (plan.rel[1] - plan.rel[0]):.0f} mm wide -> {len(plan.patches)} patch(es) "
                 f"{[round(b - a, 3) for a, b, _ in plan.patches]} m")
        self.summary["surveys"][str(mid)].update(standoff=plan.D, reach_table=plan.table,
                                                 patches=[(a, b) for a, b, _ in plan.patches])
        return plan

    # ---- 4 dock
    def dock(self, task):
        """Pre-dock marker measurement -> dock pose relative to the marker -> walk, settle, correct until within
        DOCK_TOL (the policy stops ~5 cm short: aim past by what is still missing). Returns (patch, dock_err) or None."""
        mid, I = task.mid, task.info
        plan, surf = I["plan"], I["surf"]
        pose_of = lambda f: stance_pose(f, plan.w_ref, plan.D, I["s"])
        xy, _ = self.pose()
        dock = pose_of(self.frame_est(mid))
        if np.linalg.norm(xy - dock[:2]) < 1.2:
            self.line_to(dock, f"[4 dock] {task.key}: sidestep")
        else:
            self.go_pose(dock, f"[4 dock] {task.key}", approach=0.6)
        self.reloc_on = False          # from here the work is placed relative to the marker measured from the stance
        f_meas = None
        for k in range(3):             # the marker barely fits the head camera at the stance: measure it a bit back
            self.settle()
            pre = pose_of(f_meas or self.frame_est(mid))
            pre[:2] += (PRE_DOCK + 0.12 * k) * np.array([-np.cos(pre[2]), -np.sin(pre[2])])
            xy, yaw = self.pose()
            if np.linalg.norm(xy - pre[:2]) > 0.045 or abs(wrap(yaw - pre[2])) > np.radians(3.0):
                self.line_to(pre, f"[4 dock] {task.key}: pre-dock", tol=(0.045, 2.5))
                self.settle()
            f_new = self.fresh_frame(mid, 1.0, fallback=False)
            if f_new is not False:
                f_meas = f_new
                break
        if f_meas is None:
            self.log(f"[4 dock] {task.key}: marker not measured from the pre-dock pose - using its map estimate")
            f_meas = self.frame_est(mid)
        dock = pose_of(f_meas)
        aim = dock.copy()
        for k in range(3):
            self.line_to(aim, f"[4 dock] {task.key}: dock", tol=(0.045, 2.5) if k == 0 else (0.02, 1.5))
            self.settle()
            xy, yaw = self.pose()
            e, eyaw = dock[:2] - xy, float(wrap(dock[2] - yaw))
            if np.linalg.norm(e) < DOCK_TOL[0] and abs(np.degrees(eyaw)) < DOCK_TOL[1]:
                break
            aim[:2] += e
            aim[2] += eyaw
        task.pose = dock
        f_est = self.fresh_frame(mid, 1.0, fallback=f_meas)
        xy, yaw = self.pose()
        self.log(f"[4 dock] {task.key}: at the stance {1000 * np.linalg.norm(xy - pose_of(f_est)[:2]):.0f} mm / "
                 f"{np.degrees(abs(wrap(yaw - pose_of(f_est)[2]))):.1f} deg from the marker-relative dock")
        patch = SurfacePatch(f_est, surf, I["u0"], I["u1"]).mapped(self.odo.to_true, self.odo.vec_to_true)
        sp, _ = reachable_span(self.cfg, self.ctrl, self.d.qpos.copy(), patch)
        if sp is None or sp[1] - sp[0] < 0.03:
            self.log(f"[4 dock] {task.key}: nothing reachable from this stance")
            return None
        rest = [(a, b) for a, b in ((I["u0"], sp[0]), (sp[1], I["u1"])) if b - a > 0.03]
        intended = stance_pose(self._true_frame(mid), plan.w_ref, plan.D, I["s"])          # scoring only
        R = self.d.xmat[self.ix.base].reshape(3, 3)
        dock_err = (1000 * float(np.linalg.norm(self.d.qpos[:2] - intended[:2])),
                    float(np.degrees(abs(wrap(np.arctan2(R[1, 0], R[0, 0]) - intended[2])))))
        return dict(patch=patch.sub(*sp) if rest else patch, rest=rest, f_est=f_est, dock_err=dock_err)

    # ---- 5 work
    def work(self, task, docked):
        try:
            return self.trace(task, docked["patch"], docked["dock_err"])     # v8: RL -> WBC + arm admittance -> RL
        finally:
            self.reloc_on = True

    # ---- 6 check
    def check(self, task, docked, result):
        """From the robot's own measurements: contact held (force sensor) -> else redo once; parts out of reach from the
        actual stance -> new patches (at most MAX_REMAINDERS deep). Returns the follow-up patches."""
        I, out = task.info, []
        if result != "done" and I.get("redo", 0) > 0 and docked is not None:
            self.log(f"[6 check] {task.key}: contact not held - redo once")
            out.append(Task(task.key + "x", task.mid, "patch", task.pose, task.mid, info=dict(I, redo=0)))
        if docked is not None and docked["rest"] and I["left"] > 0:
            lo, hi = I["plan"].rel
            for j, (a, b) in enumerate(docked["rest"]):
                s = (a + b) / 2 - (lo + hi) / 2
                key = f"{task.key}r{j}"
                out.append(Task(key, task.mid, "patch", stance_pose(docked["f_est"], I["plan"].w_ref, I["plan"].D, s),
                                task.mid, info=dict(I, u0=a, u1=b, s=s, left=I["left"] - 1, redo=1)))
                self.log(f"[6 check] {task.key}: {b - a:.3f} m out of reach from this stance -> patch {key}")
        if result == "done" and not out:
            self.log(f"[6 check] {task.key}: ok")
        return out

    # v8 re-fits the marker plate from depth when it is too close to decode; a curved sticker is not a plate and is only
    # partly seen at the stance (the fit was biased up to ~4 deg / 2 cm, the pre-dock measurement is within 1 deg / 2 cm)
    def refine_on_plate(self, f, mid, pts):
        return f


def plot_explore(ms, path_png):
    """Map + which surface cells the camera looked at + viewpoints + walked path + markers in the order worked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, ex = ms.world, ms.explorer
    fig, ax = plt.subplots(figsize=(15, 4.2))
    g = ex.grid
    x0, x1, y0, y1 = w.bounds
    ax.imshow(g.occ.T, origin="lower", extent=[x0, x1, y0, y1], cmap="Greys", vmax=1.6)
    st = w.stairs
    ax.add_patch(plt.Rectangle((st.x_start, st.y_lo), st.x_end - st.x_start, st.y_hi - st.y_lo, color="#bf8700", alpha=0.25))
    c = ex.w >= 1
    ax.scatter(ex.xy[c, 0], ex.xy[c, 1], s=3, c=np.where(ex.seen[c], "#2da44e", "#cf222e"))
    tr = np.array([(r[1], r[2]) for r in ms.walk_log])
    if len(tr):
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#1f6feb", lw=1.0, label="walked")
    for v in ms.viewpoints:
        ax.plot(*v, "*", color="#bf8700", ms=13, mec="k")
    for p in ms.looked_at:
        ax.plot(*p, "o", mfc="none", mec="#8250df", ms=10)
    for mid in ms.work_orders:
        p, R = w.true_marker(mid)
        ax.annotate("", p[:2] + 0.5 * R[:2, 2], p[:2], arrowprops=dict(arrowstyle="->", lw=1.5))
        k = ms.order_worked.index(mid) + 1 if mid in ms.order_worked else None
        lab = f" {mid} (#{k}, {ms.status.get(mid)})" if k else f" {mid} ({ms.status.get(mid, 'not found')})"
        ax.annotate(lab, p[:2], fontsize=8, fontweight="bold", color="k" if k else "#cf222e")
    ax.plot([], [], "*", color="#bf8700", mec="k", label="exploration viewpoint")
    ax.plot([], [], "o", mfc="none", mec="#8250df", label="look-around")
    ax.scatter([], [], s=8, c="#2da44e", label="surface looked at"); ax.scatter([], [], s=8, c="#cf222e", label="not")
    ax.plot(*w.start[:2], "go", ms=8, label="start")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right", ncol=3)
    ax.set_title(ms.banner, fontsize=9)
    fig.tight_layout(); fig.savefig(path_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="work_as_found", choices=["work_as_found", "map_first"])
    ap.add_argument("--policy", default="moe_cts", choices=["moe_cts", "symmetry"])
    ap.add_argument("--terrain-decimate", type=int, default=1)
    ap.add_argument("--panel-seed", type=int, default=4, help="shape of the random wavy panel")
    ap.add_argument("--seed", type=int, default=1, help="odometry + depth noise seed")
    ap.add_argument("--range", type=float, default=3.5, help="range at which a surface counts as looked at [m]")
    ap.add_argument("--max-marker-dist", type=float, default=4.0, help="ignore marker sightings farther than this [m]")
    ap.add_argument("--max-time", type=float, default=1200.0, help="sim time limit [s]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="live 3-D window (macOS: mjpython); Ctrl-C or close to stop")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "curved_mission")
    os.makedirs(out, exist_ok=True)
    ms = CurvedMission(args)
    ms.banner = "TASK INCOMPLETE"
    c0, err = time.time(), None
    try:
        ms.run()
    except RuntimeError as ex:
        err = str(ex)
        ms.banner = f"ABORTED: {ex}"
        ms.log(ms.banner)
    except KeyboardInterrupt:
        err = "stopped by user"
        ms.banner = "STOPPED BY USER"
        ms.log(ms.banner)
    if ms.viewer is not None:
        ms.viewer.close()
    S = ms.summary
    tilt = np.array([r[3] for r in ms.walk_log]) if ms.walk_log else np.zeros(1)
    oe = np.array([(r[1], r[2]) for r in ms.odo_log]) if ms.odo_log else np.zeros((1, 2))
    S.update(banner=ms.banner, strategy=args.strategy, order_found=list(ms.status), order_worked=ms.order_worked,
             marker_results={str(k): v for k, v in ms.marker_results.items()},
             decisions=ms.decisions,
             explore=dict(viewpoints=[np.round(v, 2).tolist() for v in ms.viewpoints], n_explore=ms.n_explore,
                          n_look_around=ms.n_observe, surface_looked_at=ms.explorer.coverage()),
             paint={str(k): v.metrics() for k, v in ms.canvases.items()},
             localisation=dict(err_max_m=float(oe[:, 0].max()), err_mean_m=float(oe[:, 0].mean()),
                               err_final_m=float(oe[-1, 0]), heading_err_max_deg=float(np.abs(oe[:, 1]).max()),
                               map_matching_updates=ms.localiser.n_updates, marker_corrections=len(ms.odo.corrections)),
             done=ms.done, failed=ms.failed, aborted=err, sim_time_s=float(ms.d.time), wall_time_s=time.time() - c0,
             walking=dict(max_tilt_deg=float(tilt.max()), distance_m=float(np.sum(np.linalg.norm(np.diff(
                 np.array([(r[1], r[2]) for r in ms.walk_log]), axis=0), axis=1))) if len(ms.walk_log) > 1 else 0.0),
             events=ms.events, policy=args.policy, seed=args.seed, panel_seed=args.panel_seed)
    json.dump(S, open(os.path.join(out, f"summary{args.tag}.json"), "w"), indent=2, default=float)
    plot_map(ms, os.path.join(out, f"map{args.tag}.png"))
    plot_explore(ms, os.path.join(out, f"explore{args.tag}.png"))
    plot_paint(ms, os.path.join(out, f"paint{args.tag}.png"))
    if ms.video is not None and ms.video.frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"mission{args.tag}.mp4"), ms.video.frames, fps=25, quality=7, macro_block_size=1)
    print("\n" + ms.banner)
    print(json.dumps({k: S[k] for k in ("order_found", "order_worked", "marker_results", "explore", "paint",
                                        "localisation", "aborted", "sim_time_s")}, indent=1, default=float))
    sys.exit(0 if ms.banner.startswith("TASK COMPLETE") else 1)


if __name__ == "__main__":
    main()
