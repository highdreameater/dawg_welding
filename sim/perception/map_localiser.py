"""Localisation against the known map: depth-scan matching to the map's signed distance field.

The robot knows the map (walls, obstacles, workpieces as footprints). Every scan of the head depth camera is a set
of points on those surfaces. Placed with the current pose estimate, the points should lie on the footprint edges;
the pose correction that pulls them there is the measurement:

    map      : footprints rasterised at 2 cm, everything outside the walls occupied; signed distance field SD
               (> 0 in free space, < 0 inside obstacles, 0 on their edges) and its gradient. Only objects whose
               footprint is their shape at every height (walls, crates, barrels, columns) are matched: a 2-D footprint
               of a pipe on saddles, a tank on a pedestal or a panel on a wall is wrong at most heights, so depth
               points around those (`exclude` footprints + 15 cm) are dropped instead of being pulled onto it
    points   : depth points in a height band above the floor relief and the stair treads and below the wall top
               (the robot's own body is invisible to the camera)
    match    : Gauss-Newton on (dx, dy, dyaw) of the estimate (rotation about the robot), residual = SD(point),
               Huber-weighted, points > 0.3 m off ignored (things not on the map)
    geometry : only directions the scan constrains are corrected (eigen-decomposition of the normal matrix) - in a
               corridor with only the side walls in view, the position along the corridor is left alone
    fusion   : a fraction `gain` of the correction per scan (several scans average the depth noise), bounded

    loc = MapLocaliser(world)
    dp, dth = loc.correct(odo, pts_est)     # pts_est: depth points mapped with the current estimate
"""
import numpy as np
from scipy import ndimage

from perception.odometry import rot2, wrap


class MapLocaliser:
    def __init__(self, world, res=0.02, band=(0.30, 1.35), huber=0.05, outlier=0.30, gain=0.5, max_step=(0.15, 3.0),
                 min_points=150, min_eig=40.0, solid=None, exclude=(), exclude_margin=0.15):
        self.res, self.band, self.huber, self.outlier, self.gain = res, band, huber, outlier, gain
        self.max_dp, self.max_dth = max_step[0], np.radians(max_step[1])
        self.min_points, self.min_eig = min_points, min_eig
        x0, x1, y0, y1 = world.bounds
        m = 0.4
        self.origin = np.array([x0 - m, y0 - m])
        nx, ny = int(np.ceil((x1 - x0 + 2 * m) / res)), int(np.ceil((y1 - y0 + 2 * m) / res))
        cx = self.origin[0] + (np.arange(nx) + 0.5) * res
        cy = self.origin[1] + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        def raster(fp, grow=0.0):
            if fp[0] == "box":
                _, bx, by, hx, hy, yaw = fp
                c, s = np.cos(yaw), np.sin(yaw)
                lx, ly = c * (X - bx) + s * (Y - by), -s * (X - bx) + c * (Y - by)
                return (np.abs(lx) <= hx + grow) & (np.abs(ly) <= hy + grow)
            _, bx, by, r = fp
            return (X - bx) ** 2 + (Y - by) ** 2 <= (r + grow) ** 2
        occ = (X < x0) | (X > x1) | (Y < y0) | (Y > y1)                     # walls: everything outside the bounds
        for fp in (world.footprints() if solid is None else solid):
            occ |= raster(fp)
        self.ignore = np.zeros_like(occ)
        for fp in exclude:
            self.ignore |= raster(fp, exclude_margin)
        self.sd = (ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ)) * res
        self.gx, self.gy = np.gradient(ndimage.gaussian_filter(self.sd, 0.7), res)
        self.n_updates, self.log = 0, []

    def _sample(self, P):
        """Bilinear SD and gradient at points P (N, 2)."""
        u = (P - self.origin) / self.res - 0.5
        i0 = np.clip(np.floor(u).astype(int), 0, np.array(self.sd.shape) - 2)
        f = np.clip(u - i0, 0, 1)
        out = []
        for F in (self.sd, self.gx, self.gy):
            a, b = F[i0[:, 0], i0[:, 1]], F[i0[:, 0] + 1, i0[:, 1]]
            c, d = F[i0[:, 0], i0[:, 1] + 1], F[i0[:, 0] + 1, i0[:, 1] + 1]
            out.append((a * (1 - f[:, 0]) + b * f[:, 0]) * (1 - f[:, 1]) + (c * (1 - f[:, 0]) + d * f[:, 0]) * f[:, 1])
        return out

    def match(self, P, centre, iters=8):
        """Pose increment (dx, dy, dth) about `centre` that moves the points P (N, 2, estimated frame) onto the map,
        restricted to the constrained directions. Returns (delta, info) or (None, info)."""
        x = np.zeros(3)
        info = dict(points=len(P))
        for _ in range(iters):
            Q = centre + (P - centre) @ rot2(x[2]).T + x[:2]
            r, gx, gy = self._sample(Q)
            ok = np.abs(r) < self.outlier
            if ok.sum() < self.min_points:
                info["reason"] = f"{ok.sum()} points on the map"
                return None, info
            rel = Q[ok] - centre
            J = np.stack([gx[ok], gy[ok], -gx[ok] * rel[:, 1] + gy[ok] * rel[:, 0]], 1)
            w = np.where(np.abs(r[ok]) < self.huber, 1.0, self.huber / np.abs(r[ok]))
            H = J.T @ (w[:, None] * J)
            g = J.T @ (w * r[ok])
            ev, V = np.linalg.eigh(H)
            keep = ev > self.min_eig
            step = -V[:, keep] @ ((V[:, keep].T @ g) / ev[keep]) if keep.any() else np.zeros(3)
            x += step
            if np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-5:
                break
        info.update(inliers=int(ok.sum()), rms_mm=float(1000 * np.sqrt(np.mean(r[ok] ** 2))),
                    constrained=int(keep.sum()), eig=ev.round(1).tolist())
        return x, info

    def correct(self, odo, pts_est):
        """Apply a (gain-scaled, bounded) map-matching correction to the odometry estimate. pts_est: 3-D depth points
        placed with the current estimate. Returns (dp, dth) applied or None."""
        P = pts_est[(pts_est[:, 2] > self.band[0]) & (pts_est[:, 2] < self.band[1])][:, :2]
        if self.ignore.any() and len(P):
            ij = np.clip(((P - self.origin) / self.res).astype(int), 0, np.array(self.ignore.shape) - 1)
            P = P[~self.ignore[ij[:, 0], ij[:, 1]]]
        if len(P) < self.min_points:
            return None
        if len(P) > 4000:
            P = P[np.random.default_rng(self.n_updates).choice(len(P), 4000, replace=False)]
        d, info = self.match(P, odo.p.copy())
        if d is None:
            return None
        dp, dth = d[:2], d[2]                      # rotation is about the robot: the robot moves by the translation only
        if np.linalg.norm(dp) > 0.6 or abs(dth) > np.radians(15):          # implausible: wrong match, ignore
            return None
        dp = self.gain * dp
        dth = self.gain * dth
        n = np.linalg.norm(dp)
        if n > self.max_dp:
            dp *= self.max_dp / n
        dth = float(np.clip(dth, -self.max_dth, self.max_dth))
        odo.p = odo.p + dp
        odo.yaw = wrap(odo.yaw + dth)
        self.n_updates += 1
        self.log.append(dict(dp=dp.round(4).tolist(), dth_deg=round(float(np.degrees(dth)), 3), **info))
        return dp, dth
