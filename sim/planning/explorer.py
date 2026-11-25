"""Exploration for markers: which obstacle surfaces has the camera already seen well enough to read a marker on
them, and where should the robot look next.

Markers sit on obstacle surfaces, so exploration is over the *surfaces* of the known map (not over unknown space -
the map is known, the markers are not):
    * surface cells  = occupied grid cells next to free space, with an outward normal (signed-distance gradient)
    * weight         = 1 on curved objects (cylinders, pipes, ellipsoids, panels: markers are stuck on curved
                       surfaces), `flat_weight` on boxes and walls, 0 on stairs
    * seen           = a camera frame had the cell inside its horizontal field of view, within `range_m`, in line of
                       sight on the map (stairs are low and do not block the view), and the surface faced the camera
                       (cos >= min_cos) - i.e. a marker there would have been detected
    * next viewpoint = free, reachable lattice point maximising (unseen weight visible in a 360 deg turn) *
                       exp(-path distance / dist_scale); None when nothing worth looking at is left

    ex = Explorer(grid, world)
    ex.mark_seen(cam_xy, cam_yaw, hfov)        # after every camera frame
    vp = ex.next_viewpoint(robot_xy)           # dict(xy, gain, dist) or None
"""
from collections import deque

import numpy as np
from scipy import ndimage


def _raster(grid, fp):
    cx = grid.origin[0] + (np.arange(grid.shape[0]) + 0.5) * grid.res
    cy = grid.origin[1] + (np.arange(grid.shape[1]) + 0.5) * grid.res
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    if fp[0] == "box":
        _, bx, by, hx, hy, yaw = fp
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c * (X - bx) + s * (Y - by), -s * (X - bx) + c * (Y - by)
        return (np.abs(lx) <= hx + 0.5 * grid.res) & (np.abs(ly) <= hy + 0.5 * grid.res)
    _, bx, by, r = fp
    return (X - bx) ** 2 + (Y - by) ** 2 <= (r + 0.5 * grid.res) ** 2


class Explorer:
    def __init__(self, grid, world, range_m=3.5, min_cos=0.3, flat_weight=0.15, lattice=0.30, min_gain=4.0,
                 dist_scale=6.0):
        self.grid, self.range_m, self.min_cos = grid, range_m, min_cos
        self.min_gain, self.dist_scale = min_gain, dist_scale
        occ = grid.occ
        weight = np.full(grid.shape, flat_weight)
        stairs = np.zeros(grid.shape, bool)
        for ob in world.map_objects():
            r = _raster(grid, ob["footprint"]) & occ
            if ob["kind"] == "stairs":
                stairs |= r
            elif ob["curved"]:
                weight[r] = 1.0
        weight[stairs] = 0.0
        self.block = occ & ~stairs                                   # what blocks the line of sight
        surf = occ & ndimage.binary_dilation(~occ, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        sd = ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)
        gx, gy = np.gradient(ndimage.gaussian_filter(sd, 1.0))
        nrm = np.hypot(gx, gy) + 1e-9
        self.cells = np.argwhere(surf & (weight > 0))               # (K, 2) cell indices
        self.xy = grid.origin + (self.cells + 0.5) * grid.res
        self.nrm = np.stack([gx, gy], -1)[surf & (weight > 0)] / nrm[surf & (weight > 0)][:, None]
        self.w = weight[surf & (weight > 0)]
        self.seen = np.zeros(len(self.cells), bool)
        step = max(1, int(round(lattice / grid.res)))
        lat = np.zeros(grid.shape, bool)
        lat[step // 2::step, step // 2::step] = True
        self.lattice = np.argwhere(lat & grid.free)
        self.visited_viewpoints = []

    # ------------------------------------------------------------------ visibility
    def _visible(self, p, idx):
        """Which surface cells `idx` are seen from point p (within range, facing it, line of sight)."""
        if len(idx) == 0:
            return np.zeros(0, bool)
        v = p - self.xy[idx]
        d = np.linalg.norm(v, axis=1)
        ok = (d <= self.range_m) & (np.sum(self.nrm[idx] * v, 1) >= self.min_cos * d)
        idx_ok = idx[ok]
        if len(idx_ok):
            g = self.grid
            a = (p - g.origin) / g.res                                  # in cell units
            b = self.cells[idx_ok] + 0.5
            L = np.linalg.norm(b - a, axis=1)
            S = int(np.ceil(2 * L.max())) + 1
            t = np.linspace(0, 1, S)[None, :] * np.clip(1 - 0.9 / np.maximum(L, 1e-6), 0, 1)[:, None]
            q = a[None, None, :] + t[:, :, None] * (b - a)[:, None, :]
            qi = np.clip(np.floor(q).astype(int), 0, np.array(g.shape) - 1)
            blocked = self.block[qi[..., 0], qi[..., 1]]
            blocked[:, 0] = False                                       # the robot's own cell
            ok[np.flatnonzero(ok)] = ~blocked.any(1)
        return ok

    def mark_seen(self, cam_xy, cam_yaw, hfov):
        p = np.asarray(cam_xy, float)
        v = self.xy - p
        ang = np.arctan2(v[:, 1], v[:, 0]) - cam_yaw
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        idx = np.flatnonzero((np.abs(ang) <= 0.45 * hfov) & ~self.seen)
        vis = self._visible(p, idx)
        self.seen[idx[vis]] = True
        return int(vis.sum())

    # ------------------------------------------------------------------ next viewpoint
    def _reach(self, start_xy):
        """Grid distance [m] (8-connected BFS steps) over free cells from the robot's cell; inf = unreachable."""
        g = self.grid
        D = np.full(g.shape, np.inf)
        s = g.to_cell(start_xy)
        if not g.free[s]:
            r = np.argwhere(g.free)
            s = tuple(r[np.argmin(np.sum((r - s) ** 2, 1))])
        D[s] = 0.0
        q = deque([s])
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        free = g.free
        nx, ny = g.shape
        while q:
            c = q.popleft()
            dc = D[c] + 1
            for dx, dy in nb:
                n = (c[0] + dx, c[1] + dy)
                if 0 <= n[0] < nx and 0 <= n[1] < ny and free[n] and D[n] == np.inf:
                    D[n] = dc
                    q.append(n)
        return D * g.res

    def gain(self, p):
        idx = np.flatnonzero(~self.seen)
        return float(self.w[idx][self._visible(np.asarray(p, float), idx)].sum())

    def next_viewpoint(self, robot_xy):
        D = self._reach(robot_xy)
        idx = np.flatnonzero(~self.seen)
        best = None
        for c in self.lattice:
            dist = D[c[0], c[1]]
            if not np.isfinite(dist):
                continue
            p = self.grid.to_xy(c)
            vis = self._visible(p, idx)
            g = float(self.w[idx][vis].sum())
            if g < self.min_gain:
                continue
            score = g * np.exp(-dist / self.dist_scale)
            if best is None or score > best["score"]:
                best = dict(xy=p, gain=g, dist=float(dist), score=float(score))
        if best is not None:
            self.visited_viewpoints.append(best["xy"])
        return best

    def coverage(self):
        """Seen fraction of the curved surface (weight 1) cells."""
        c = self.w >= 1.0
        return float(self.seen[c].mean()) if c.any() else 1.0
