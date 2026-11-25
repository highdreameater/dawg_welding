"""Visual record of the contact work: a force-shaded "paint" trail on the *real* surfaces (intensity brush).

Wherever the contact tip actually touches the real surface, the brush (2.2 cm wide, strokes 2 cm apart) stamps the
cells under it with the measured contact force F. Shade = intensity of the force actually applied:
    F -> 0   light tint (pressing too little)      F = F_des   the work colour      F >= 2 F_des  dark (too hard)
A later stroke over the same cell overwrites it (it shows the force of the last pass, not an accumulation).

SurfaceCanvas: one generic canvas for any work surface. A grid over the work rectangle in its (true) area frame
(u right, v up, z out); the real surface under every cell is found by ray-casting the scene along -z, once, when the
canvas is made. This is ground truth for drawing and scoring only - the robot never sees it.
"""
import mujoco
import numpy as np


class SurfaceCanvas:
    def __init__(self, m, d, o, ex, ey, ez, size, target_rect, cell=0.006, color=(0.85, 0.12, 0.10), reach=0.5):
        """o: lower-left corner of the canvas on the area plane; size: (U, V) [m]; target_rect: (u0, v0, u1, v1)."""
        self.o, self.ex, self.ey, self.ez = (np.asarray(a, float) for a in (o, ex, ey, ez))
        self.cell, self.color = cell, np.asarray(color, float)
        ni, nj = int(round(size[0] / cell)), int(round(size[1] / cell))
        self.F = np.full((ni, nj), np.nan)                # measured force of the last stroke over each cell [N]
        self.f_des = None
        U, V = np.meshgrid((np.arange(ni) + 0.5) * cell, (np.arange(nj) + 0.5) * cell, indexing="ij")
        u0, v0, u1, v1 = target_rect
        self.target = (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)
        # true surface under each cell: ray from `reach` in front of the area plane along -ez (world geoms only)
        start = self.o + U[..., None] * self.ex + V[..., None] * self.ey + reach * self.ez
        n = ni * nj
        dist, gid = np.zeros(n), np.zeros(1, np.int32)
        grp = np.array([0, 1, 0, 0, 0, 0], np.uint8)
        for k, s in enumerate(start.reshape(-1, 3)):     # mj_multiRay needs one origin: cast cell by cell
            dist[k] = mujoco.mj_ray(m, d, s, -self.ez, grp, 1, -1, gid)
        hit = (dist > 0) & (dist < 2 * reach)
        P = start.reshape(-1, 3) - np.where(hit, dist, reach)[:, None] * self.ez
        self.P = P.reshape(ni, nj, 3)
        gu = np.gradient(self.P, axis=0)
        gv = np.gradient(self.P, axis=1)
        N = np.cross(gu, gv)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
        N[(N @ self.ez) < 0] *= -1
        self.N = N
        self.hit = hit.reshape(ni, nj)

    def to_ij(self, p):
        rel = np.asarray(p) - self.o
        return np.array([rel @ self.ex, rel @ self.ey]) / self.cell - 0.5

    def stamp(self, p, force, f_des, width=0.022):
        self.f_des = f_des
        if force < 0.3:
            return
        ij = self.to_ij(p)
        r = width / 2 / self.cell
        ni, nj = self.F.shape
        k = int(np.ceil(r))
        i0, j0 = int(round(ij[0])), int(round(ij[1]))
        ia, ib, ja, jb = max(0, i0 - k), min(ni, i0 + k + 1), max(0, j0 - k), min(nj, j0 + k + 1)
        if ia >= ib or ja >= jb:
            return
        ii, jj = np.mgrid[ia:ib, ja:jb]
        inside = (ii - ij[0]) ** 2 + (jj - ij[1]) ** 2 <= r ** 2
        self.F[ii[inside], jj[inside]] = force

    def metrics(self):
        t, f = self.target, self.f_des or 5.0
        painted = ~np.isnan(self.F)
        Ft = self.F[t & painted]
        return dict(coverage=float(np.mean(painted[t])),
                    force_mean=float(Ft.mean()) if len(Ft) else None,
                    force_p5_p95=[float(np.percentile(Ft, 5)), float(np.percentile(Ft, 95))] if len(Ft) else None,
                    within_20pct=float(np.mean(np.abs(Ft / f - 1) <= 0.2)) if len(Ft) else None,
                    outside_target=float(painted[~t].sum() / max(painted.sum(), 1)))

    def shade(self, F):
        s = np.clip(F / (2.0 * (self.f_des or 5.0)), 0.0, 1.0)
        light = 0.2 * self.color + 0.8 * np.ones(3)
        dark = 0.3 * self.color
        if s < 0.5:
            return light + (s / 0.5) * (self.color - light)
        return self.color + ((s - 0.5) / 0.5) * (dark - self.color)

    def draw(self, scn, thick=0.0012):
        h = self.cell / 2 * 1.02
        for i, j in zip(*np.nonzero(~np.isnan(self.F))):
            if scn.ngeom >= scn.maxgeom:
                break
            n = self.N[i, j]
            u = self.ex - (self.ex @ n) * n
            u /= np.linalg.norm(u)
            mat = np.stack([u, np.cross(n, u), n], 1)
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([h, h, thick]),
                                self.P[i, j] + (thick + 0.0008) * n, mat.ravel(),
                                np.array([*self.shade(self.F[i, j]), 1.0], np.float32))
            scn.ngeom += 1
