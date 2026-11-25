"""Planning on the known map for the ArUco work-order mission.

Map = sim/aruco_world.py walls + obstacles + the staircase. The robot is localised in it (no SLAM).

Paths: occupancy grid (5 cm) + A* (sim/planning/grid_planner.py: clearance cost, shortcut, Chaikin smoothing).
The staircase is a *portal*: its cells are blocked in the grid and it is crossed only along one straight edge
(entry -> exit, perpendicular to the steps, at the y with the most clearance on both sides), because the robot must
climb straight. A route that changes sides is A*(start -> entry) + stairs edge + A*(exit -> goal); the stairs edge
costs its length + STAIRS_PENALTY.

Obstacles the robot discovers (fitted pipes, sign posts) are added to the grid at run time.

Task queue: given the robot pose and the pending task poses, the visiting order minimises the total travel cost
(A* path lengths, brute force over permutations - fine for <= 7 pending tasks). The robot goes task -> task
directly (no return to base); the order is recomputed whenever a new marker is confirmed.
"""
import itertools

import numpy as np

from planning.grid_planner import OccupancyGrid, plan_path, resample

STAIRS_PENALTY = 2.0        # [m] equivalent: climbing is slower and riskier than flat walking
PORTAL_GAP = 0.70           # entry / exit points this far from the first / last riser


class MapView:
    """What the planner sees: the world's footprints + the staircase as a box + discovered obstacles."""

    def __init__(self, world):
        self.world, self.bounds, self.extra = world, world.bounds, []

    def footprints(self):
        st = self.world.stairs
        fps = self.world.footprints() + list(self.extra)
        fps.append(("box", (st.x_start + st.x_end) / 2, (st.y_lo + st.y_hi) / 2, (st.x_end - st.x_start) / 2,
                    (st.y_hi - st.y_lo) / 2, 0.0))
        return fps


class PortalPlanner:
    def __init__(self, world, robot_radius=0.45, res=0.05):
        self.world, self.view, self.robot_radius, self.res = world, MapView(world), robot_radius, res
        self._build()

    def _build(self):
        self.grid = OccupancyGrid(self.view, res=self.res, robot_radius=self.robot_radius, clearance=0.9)
        st = self.world.stairs
        xa, xb = st.x_start - PORTAL_GAP, st.x_end + PORTAL_GAP
        ys = np.arange(st.y_lo + 0.5, st.y_hi - 0.5, 0.05)
        clear = [min(self.grid.clearance_at((xa, y)), self.grid.clearance_at((xb, y))) for y in ys]
        y = float(ys[int(np.argmax(clear))])
        self.portal = (np.array([xa, y]), np.array([xb, y]))
        self._cache = {}

    def add_obstacle(self, fp):
        self.view.extra.append(fp)
        self._build()

    def side(self, xy):
        st = self.world.stairs
        return 0 if xy[0] < (st.x_start + st.x_end) / 2 else 1

    def on_stairs(self, xy):
        st = self.world.stairs
        return st.x_start - 0.15 <= xy[0] <= st.x_end + 0.15

    def on_platform(self, xy, margin=0.25):
        """Inside the stair platform (flat top between the up and down flights), `margin` from its edges."""
        st = self.world.stairs
        return (st.x_start + st.n * st.run + margin <= xy[0] <= st.x_end - st.n * st.run - margin
                and st.y_lo + margin <= xy[1] <= st.y_hi - margin)

    def platform_center(self):
        st = self.world.stairs
        return np.array([(st.x_start + st.x_end) / 2, self.portal[0][1]])

    def ground_at(self, xy, floor_z):
        """Map ground height at xy: the staircase is in the map; elsewhere the floor (floor_z, measured)."""
        st = self.world.stairs
        if st.y_lo <= xy[1] <= st.y_hi:
            h = st.height(xy[0])
            if h is not None:
                return h
        return floor_z

    def stance_ground(self, xy, yaw, floor_z, half_len=0.25, half_wid=0.16):
        """Ground under a stance (four feet at +-half_len along, +-half_wid across the heading), or None if the feet
        would not all stand on the same level (straddling a step / the platform edge)."""
        c, s = np.cos(yaw), np.sin(yaw)
        hs = [self.ground_at(np.asarray(xy) + a * np.array([c, s]) + b * np.array([-s, c]), floor_z)
              for a in (-half_len, half_len) for b in (-half_wid, half_wid)]
        return hs[0] if max(hs) - min(hs) < 0.01 else None

    def plan(self, start, goal, goal_yaw=None, approach=0.8):
        """Returns (path (N, 2), info) with info["stairs"] = (i0, i1) index range on the stairs, or None."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        s0, s1 = self.side(start), self.side(goal)
        if s0 == s1:
            path, info = plan_path(self.grid, start, goal, goal_yaw, approach=approach)
            info["stairs"] = None
            return path, info
        a, b = self.portal if s0 == 0 else self.portal[::-1]
        yaw_in = 0.0 if s0 == 0 else np.pi
        if np.linalg.norm(start - a) > 0.1:
            p1, i1 = plan_path(self.grid, start, a, yaw_in, approach=0.6)
        else:
            p1, i1 = np.array([start, a]), dict(length=float(np.linalg.norm(start - a)))
        cross = resample(np.array([a, b]), 0.05)
        p2, i2 = plan_path(self.grid, b, goal, goal_yaw, approach=approach)
        path = np.vstack([p1, cross[1:], p2[1:]])
        k0 = len(p1) - 1
        info = dict(length=i1["length"] + float(np.linalg.norm(b - a)) + i2["length"], stairs=(k0, k0 + len(cross) - 1),
                    min_clearance=min(i1.get("min_clearance", 9), i2["min_clearance"]),
                    raw=np.vstack([i1.get("raw", p1), i2["raw"]]))
        return path, info

    def cost(self, start, goal, goal_yaw=None):
        key = (tuple(np.round(start, 1)), tuple(np.round(goal, 2)), None if goal_yaw is None else round(goal_yaw, 2),
               len(self.view.extra))
        if key not in self._cache:
            try:
                _, info = self.plan(start, goal, goal_yaw)
                self._cache[key] = info["length"] + (STAIRS_PENALTY if info["stairs"] else 0.0)
            except RuntimeError:
                self._cache[key] = np.inf
        return self._cache[key]

    def order(self, start_xy, tasks):
        """tasks: list of (key, pose (x, y, yaw), group). Tasks of one group (e.g. the patches of one wall band)
        are kept consecutive and done in the order given. Returns (ordered keys, total cost)."""
        groups = {}
        for key, pose, g in tasks:
            groups.setdefault(g, []).append((key, np.asarray(pose, float)))
        best, best_c = None, np.inf
        for perm in itertools.permutations(groups):
            seq = []
            for g in perm:                                  # each group: start from the end closer to where we are
                items = groups[g]
                prev = seq[-1][1][:2] if seq else np.asarray(start_xy)
                if len(items) > 1 and (np.linalg.norm(items[-1][1][:2] - prev) < np.linalg.norm(items[0][1][:2] - prev)):
                    items = items[::-1]
                seq += items
            c, prev = 0.0, np.asarray(start_xy, float)
            for _, pose in seq:
                c += self.cost(prev, pose[:2], pose[2])
                prev = pose[:2]
                if c >= best_c:
                    break
            if c < best_c:
                best, best_c = [k for k, _ in seq], c
        return best, best_c


def patrol_waypoints(planner, start_xy, spacing=3.0, min_clear=0.8):
    """Exploration waypoints from the map alone (no hand-placed list): farthest-point sampling over the free cells
    with >= min_clear clearance until every such cell is within `spacing` of a waypoint, then ordered greedily by
    A* travel cost from the start (stairs included via the portal)."""
    g = planner.grid
    ij = np.argwhere(g.dist >= min_clear)
    cand = np.array([g.to_xy(c) for c in ij])
    d = np.linalg.norm(cand - np.asarray(start_xy), axis=1)
    pts = []
    while True:
        k = int(np.argmax(d))
        if d[k] <= spacing:
            break
        pts.append(cand[k])
        d = np.minimum(d, np.linalg.norm(cand - cand[k], axis=1))
    order, prev, left = [], np.asarray(start_xy, float), list(range(len(pts)))
    while left:
        k = min(left, key=lambda i: planner.cost(prev, pts[i]))
        if not np.isfinite(planner.cost(prev, pts[k])):
            break
        order.append(pts[k]); prev = pts[k]; left.remove(k)
    return order
