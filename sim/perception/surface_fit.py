"""Reconstruct an unknown freeform surface z = h(x, y) from a depth point cloud - no parametric shape assumed.

    1. crop to the region of interest (the stand's footprint from the map, + margin)
    2. 7.5 mm xy cells; per cell keep the points within 6 mm of the cell's highest point and take their median
       (the top surface wins over the stand's vertical sides; median rejects depth noise)
    3. one global smoothing thin-plate spline (scipy RBFInterpolator, all cells at once) through the cell heights.
       (A local RBF - `neighbors=k` - is discontinuous between neighbourhoods: ~0.1 mm ripples at ~1 cm
       wavelength that the force controller feels as 3 Hz normal / curvature noise and loses contact.)
    4. resample on a 2.5 mm grid and fit a bicubic spline: height and normals (analytic derivatives) are C2-smooth,
       which the force controller needs (a bilinear grid has piecewise-constant slopes: its curvature feed-forward
       spikes at every cell border and the contact force chatters by +-5 N); a coverage mask says where the
       surface was actually observed

    model = fit_surface(points, roi=(cx, cy, hx, hy))
    model.height(x, y), model.normal(x, y), model.covered(x, y)
"""
import numpy as np
from scipy.interpolate import RBFInterpolator, RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_dilation


class SurfaceModel:
    def __init__(self, xs, ys, H, covered, info):
        self.xs, self.ys, self.H = xs, ys, H
        self._spl = RectBivariateSpline(ys, xs, H, kx=3, ky=3)   # first coordinate = y
        kw = dict(bounds_error=False, fill_value=None)
        self._cov = RegularGridInterpolator((ys, xs), covered.astype(float), method="nearest", **kw)
        self.info = info

    def _q(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return np.stack([y.ravel(), x.ravel()], 1), x.shape

    def height(self, x, y):
        q, sh = self._q(x, y)
        return self._spl.ev(q[:, 0], q[:, 1]).reshape(sh)

    def normal(self, x, y):
        q, sh = self._q(x, y)
        hx, hy = self._spl.ev(q[:, 0], q[:, 1], dy=1), self._spl.ev(q[:, 0], q[:, 1], dx=1)
        n = np.stack([-hx, -hy, np.ones(len(q))], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return n.reshape(*sh, 3)

    def covered(self, x, y):
        q, sh = self._q(x, y)
        return (self._cov(q) > 0.5).reshape(sh)


class FlatSurface:
    """The 'no perception' belief: a flat top at the map's nominal height."""

    def __init__(self, z):
        self.z = z
        self.info = dict(kind="flat", z=z)

    def height(self, x, y):
        return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, self.z)

    def normal(self, x, y):
        sh = np.broadcast(np.asarray(x), np.asarray(y)).shape
        return np.broadcast_to(np.array([0.0, 0.0, 1.0]), (*sh, 3)).copy()

    def covered(self, x, y):
        return np.ones(np.broadcast(np.asarray(x), np.asarray(y)).shape, bool)


def fit_surface(pts, roi, margin=0.02, cell=0.0075, band=0.006, smoothing=2e-3, out_res=0.0025):
    cx, cy, hx, hy = roi
    x0, x1, y0, y1 = cx - hx - margin, cx + hx + margin, cy - hy - margin, cy + hy + margin
    p = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
    if len(p) < 200:
        raise RuntimeError(f"too few surface points in the region of interest ({len(p)})")
    ij = np.floor((p[:, :2] - [x0, y0]) / cell).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((-p[:, 2], key))                 # by cell, highest first
    key, p = key[order], p[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    cells = []
    for a, b in zip(starts, ends):
        z = p[a:b, 2]
        top = z[z > z[0] - band]
        if len(top) >= 2:
            cells.append((*np.median(p[a:b][z > z[0] - band, :2], axis=0), np.median(top)))
    cells = np.array(cells)
    rbf = RBFInterpolator(cells[:, :2], cells[:, 2], kernel="thin_plate_spline", smoothing=smoothing)
    xs = np.arange(cx - hx, cx + hx + 1e-9, out_res)
    ys = np.arange(cy - hy, cy + hy + 1e-9, out_res)
    X, Y = np.meshgrid(xs, ys)
    H = rbf(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    occ = np.zeros(X.shape, bool)
    ci = np.clip(((cells[:, 0] - xs[0]) / out_res).round().astype(int), 0, len(xs) - 1)
    cj = np.clip(((cells[:, 1] - ys[0]) / out_res).round().astype(int), 0, len(ys) - 1)
    occ[cj, ci] = True
    covered = binary_dilation(occ, iterations=4)        # within ~1 cm of an observed cell
    return SurfaceModel(xs, ys, H, covered, dict(kind="thin-plate spline", points=int(len(p)), cells=int(len(cells)),
                                                coverage=float(covered.mean())))
