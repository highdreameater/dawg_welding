"""Global path planning on a known 2-D map (no SLAM): occupancy grid -> A* with clearance cost -> shortcut + smoothing.

    grid = OccupancyGrid(world, res=0.05, robot_radius=0.50)
    path = plan_path(grid, start_xy, goal_xy)          # (N, 2) waypoints, ~5 cm apart, collision-checked

* Occupancy comes from the same obstacle list that builds the MuJoCo scene (sim/world.py), plus the walls.
* The robot is a disc of `robot_radius` for collision (configuration space = obstacles dilated by it).
* The A* edge cost adds a penalty for passing within `clearance` of obstacles, so paths keep a margin where
  there is room and only squeeze through narrow gaps when they must.
* The raw grid path is shortened with line-of-sight shortcuts, corner-cut (Chaikin) and resampled; every
  smoothing step is re-checked against the dilated map.
"""
import heapq

import numpy as np
from scipy import ndimage


class OccupancyGrid:
    def __init__(self, world, res=0.05, robot_radius=0.50, clearance=1.0):
        self.res, self.robot_radius, self.clearance = res, robot_radius, clearance
        x0, x1, y0, y1 = world.bounds
        self.origin = np.array([x0, y0])
        self.shape = (int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res)))  # (nx, ny), index [ix, iy]
        cx = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * res
        cy = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * res
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        occ = np.zeros(self.shape, bool)
        for fp in world.footprints():
            if fp[0] == "box":
                _, bx, by, hx, hy, yaw = fp
                c, s = np.cos(yaw), np.sin(yaw)
                lx, ly = c * (X - bx) + s * (Y - by), -s * (X - bx) + c * (Y - by)
                occ |= (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
            else:
                _, bx, by, r = fp
                occ |= (X - bx) ** 2 + (Y - by) ** 2 <= r ** 2
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True  # walls
        self.occ = occ
        self.dist = ndimage.distance_transform_edt(~occ) * res   # [m] to the nearest obstacle
        self.free = self.dist > robot_radius                     # configuration-space free cells

    def to_cell(self, p):
        i = np.floor((np.asarray(p, float) - self.origin) / self.res).astype(int)
        return int(np.clip(i[0], 0, self.shape[0] - 1)), int(np.clip(i[1], 0, self.shape[1] - 1))

    def to_xy(self, c):
        return self.origin + (np.asarray(c, float) + 0.5) * self.res

    def clearance_at(self, p):
        return float(self.dist[self.to_cell(p)])

    def segment_free(self, a, b, margin=0.0):
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / (0.5 * self.res))) + 1)
        for s in np.linspace(0, 1, n):
            if self.dist[self.to_cell((1 - s) * np.asarray(a) + s * np.asarray(b))] <= self.robot_radius + margin:
                return False
        return True


def _nearest_free(grid, cell, max_r=40):
    if grid.free[cell]:
        return cell
    best = None
    for r in range(1, max_r):
        for dx in range(-r, r + 1):
            for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                c = (cell[0] + dx, cell[1] + dy)
                if 0 <= c[0] < grid.shape[0] and 0 <= c[1] < grid.shape[1] and grid.free[c]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, c)
        if best:
            return best[1]
    raise RuntimeError(f"no free cell near {cell}")


def astar(grid, start_cell, goal_cell, w_clear=4.0):
    """8-connected A*. Edge cost = length * (1 + w_clear * proximity), proximity in [0, 1] below `clearance`."""
    nx, ny = grid.shape
    prox = np.clip((grid.clearance - grid.dist) / (grid.clearance - grid.robot_radius), 0.0, 1.0)
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (-1, -1, 2 ** 0.5)]
    g = np.full(grid.shape, np.inf)
    parent = -np.ones(grid.shape + (2,), int)
    g[start_cell] = 0.0
    h = lambda c: np.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])
    openq = [(h(start_cell), 0.0, start_cell)]
    closed = np.zeros(grid.shape, bool)
    while openq:
        _, gc, c = heapq.heappop(openq)
        if closed[c]:
            continue
        closed[c] = True
        if c == goal_cell:
            break
        for dx, dy, L in moves:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or closed[n] or not grid.free[n]:
                continue
            ng = gc + L * (1.0 + w_clear * prox[n])
            if ng < g[n]:
                g[n] = ng
                parent[n] = c
                heapq.heappush(openq, (ng + h(n), ng, n))
    if not np.isfinite(g[goal_cell]):
        raise RuntimeError("no path to goal")
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(tuple(parent[cells[-1]]))
    return cells[::-1], int(closed.sum())


def shortcut(grid, pts, margin):
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not grid.segment_free(pts[i], pts[j], margin):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def chaikin(grid, pts, iters=3):
    for _ in range(iters):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        new.append(pts[-1])
        new = np.array(new)
        if all(grid.segment_free(a, b) for a, b in zip(new[:-1], new[1:])):
            pts = new
        else:
            break
    return pts


def resample(pts, ds=0.05):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / ds)) + 1))
    return np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], 1)


def plan_path(grid, start_xy, goal_xy, goal_yaw=None, approach=0.8, dock_clearance=0.30, w_clear=4.0):
    """Plan start -> goal. With `goal_yaw`, the last `approach` metres are a straight line along the goal heading
    (the robot docks head-on at a work station). That docking line may come closer to obstacles than the robot
    radius - the robot stands *at* its workpiece, nose over it - but keeps `dock_clearance` (half the leg span).
    Returns (path (N,2), info dict)."""
    goal_xy = np.asarray(goal_xy, float)
    target = goal_xy
    if goal_yaw is not None and approach > 0:
        target = goal_xy - approach * np.array([np.cos(goal_yaw), np.sin(goal_yaw)])  # pre-dock point
        if not grid.free[grid.to_cell(target)]:
            raise RuntimeError(f"pre-dock point {target} is in collision")
        if not grid.segment_free(target, goal_xy, margin=dock_clearance - grid.robot_radius):
            raise RuntimeError("straight docking approach is blocked")
    elif not grid.free[grid.to_cell(goal_xy)]:
        raise RuntimeError(f"goal {goal_xy} is in collision (clearance {grid.clearance_at(goal_xy):.2f} m)")
    sc, gc = _nearest_free(grid, grid.to_cell(start_xy)), grid.to_cell(target)
    cells, expanded = astar(grid, sc, gc, w_clear)
    raw = np.array([grid.to_xy(c) for c in cells])
    raw[0], raw[-1] = start_xy, target
    # shortcuts keep a 15 cm margin beyond the robot radius where possible (tight spots keep the A* cells)
    short = shortcut(grid, raw, margin=0.15)
    smooth = chaikin(grid, short)
    if target is not goal_xy:  # append the straight docking line (checked above with dock_clearance)
        raw, short, smooth = (np.vstack([a, goal_xy]) for a in (raw, short, smooth))
    path = resample(smooth, 0.05)
    clear = np.array([grid.clearance_at(p) for p in path])
    info = dict(expanded=expanded, raw_len=float(np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1))),
                length=float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
                min_clearance=float(clear.min()), raw=raw, shortcut=short)
    return path, info
