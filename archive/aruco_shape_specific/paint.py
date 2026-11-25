"""Visual record of the contact work: a force-shaded "paint" trail on the *real* surfaces.

The paint is only a representation of the work: wherever the contact tip actually touches the real surface, the
cells under it (stroke width 2.2 cm, strokes 2 cm apart) take the measured contact force as a ratio r = F / F_des. Colour:
    r -> 0   light tint (pressing too little)      r = 1   mid tone (on target)      r >= 2  dark (pressing too hard)
A later stroke over the same cell overwrites it (it shows the force of the last pass, not an accumulation).

Canvas = a grid over a surface patch, `cell` metres:
    PlaneCanvas    : a rectangle on a wall (origin = lower-left corner, u = right, v = up, n = outward)
    CylinderCanvas : a patch of a vertical pipe (axis, radius, angle range around a reference direction, z range)
"""
import mujoco
import numpy as np


class Canvas:
    def __init__(self, shape, cell, color, target_mask):
        self.r = np.full(shape, np.nan)          # force ratio of the last stroke over each cell
        self.cell, self.color = cell, np.asarray(color, float)
        self.target = target_mask

    def _stamp(self, ij, radius_cells, ratio):
        ni, nj = self.r.shape
        k = int(np.ceil(radius_cells))
        if not np.all(np.isfinite(ij)):
            return
        i0, j0 = int(round(ij[0])), int(round(ij[1]))
        ia, ib, ja, jb = max(0, i0 - k), min(ni, i0 + k + 1), max(0, j0 - k), min(nj, j0 + k + 1)
        if ia >= ib or ja >= jb:
            return
        ii, jj = np.mgrid[ia:ib, ja:jb]
        inside = (ii - ij[0]) ** 2 + (jj - ij[1]) ** 2 <= radius_cells ** 2
        self.r[ii[inside], jj[inside]] = ratio

    def stamp(self, p, force, f_des, width=0.022):
        if force < 0.3:
            return
        self._stamp(self.to_ij(p), width / 2 / self.cell, force / f_des)

    def metrics(self):
        t = self.target
        painted = ~np.isnan(self.r)
        rt = self.r[t & painted]
        return dict(coverage=float(np.mean(painted[t])),
                    force_ratio_mean=float(rt.mean()) if len(rt) else None,
                    force_ratio_p5_p95=[float(np.percentile(rt, 5)), float(np.percentile(rt, 95))] if len(rt) else None,
                    within_20pct=float(np.mean(np.abs(rt - 1) <= 0.2)) if len(rt) else None,
                    outside_target=float(painted[~t].sum() / max(painted.sum(), 1)))

    def shade(self, r):
        s = np.clip(r / 2.0, 0.0, 1.0)
        light = 0.25 * self.color + 0.75 * np.ones(3)
        dark = 0.35 * self.color
        if s < 0.5:
            return light + (s / 0.5) * (self.color - light)
        return self.color + ((s - 0.5) / 0.5) * (dark - self.color)

    def draw(self, scn, thick=0.0012):
        P, N, U = self.cell_frames()
        R = self.r.ravel()
        h = self.cell / 2 * 1.02
        for k in np.nonzero(~np.isnan(R))[0]:
            if scn.ngeom >= scn.maxgeom:
                break
            n, u = N[k], U[k]
            mat = np.stack([u, np.cross(n, u), n], 1)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([h, h, thick]),
                                P[k] + (thick + 0.0008) * n, mat.ravel(), np.array([*self.shade(R[k]), 1.0], np.float32))
            scn.ngeom += 1


class PlaneCanvas(Canvas):
    def __init__(self, origin, u, v, n, size, target_rect, cell=0.005, color=(0.85, 0.12, 0.10)):
        """target_rect: (u0, v0, u1, v1) in metres from the origin."""
        self.o, self.u, self.v, self.n = (np.asarray(a, float) for a in (origin, u, v, n))
        ni, nj = int(round(size[0] / cell)), int(round(size[1] / cell))
        U, V = np.meshgrid((np.arange(ni) + 0.5) * cell, (np.arange(nj) + 0.5) * cell, indexing="ij")
        u0, v0, u1, v1 = target_rect
        super().__init__((ni, nj), cell, color, (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1))
        self._P = (self.o + U[..., None] * self.u + V[..., None] * self.v).reshape(-1, 3)

    def to_ij(self, p):
        rel = np.asarray(p) - self.o
        return np.array([rel @ self.u, rel @ self.v]) / self.cell - 0.5

    def cell_frames(self):
        n = len(self._P)
        return self._P, np.tile(self.n, (n, 1)), np.tile(self.u, (n, 1))


class CylinderCanvas(Canvas):
    def __init__(self, axis_xy, radius, ref_dir, theta_range, z_range, target, cell=0.005, color=(0.10, 0.35, 0.90)):
        """Vertical cylinder. theta around +z from the horizontal unit `ref_dir`; target: (theta0, z0, theta1, z1)."""
        self.c, self.rad = np.asarray(axis_xy, float), radius
        self.e1 = np.array([*ref_dir[:2], 0.0]); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross([0, 0, 1.0], self.e1)
        self.th0, self.z0 = theta_range[0], z_range[0]
        self.dth = cell / radius
        ni = int(round((theta_range[1] - theta_range[0]) / self.dth))
        nj = int(round((z_range[1] - z_range[0]) / cell))
        TH, Z = np.meshgrid(self.th0 + (np.arange(ni) + 0.5) * self.dth, self.z0 + (np.arange(nj) + 0.5) * cell, indexing="ij")
        t0, zz0, t1, zz1 = target
        super().__init__((ni, nj), cell, color, (TH >= t0) & (TH <= t1) & (Z >= zz0) & (Z <= zz1))
        Nrm = np.cos(TH)[..., None] * self.e1 + np.sin(TH)[..., None] * self.e2
        self._N = Nrm.reshape(-1, 3)
        self._P = (np.array([*self.c, 0.0]) + radius * Nrm + Z[..., None] * np.array([0, 0, 1.0])).reshape(-1, 3)
        self._U = np.cross([0, 0, 1.0], self._N)

    def to_ij(self, p):
        v = np.asarray(p[:2], float) - self.c
        th = np.arctan2(v @ self.e2[:2], v @ self.e1[:2])
        return np.array([(th - self.th0) / self.dth, (p[2] - self.z0) / self.cell]) - 0.5

    def cell_frames(self):
        return self._P, self._N, self._U
