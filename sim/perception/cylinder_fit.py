"""Detect a pipe (cylinder) in a depth point cloud: axis, radius and near end - no prior about its pose or size.

    1. crop: drop the floor (z < 3 cm); keep a region of interest around where the map says the pipe roughly is
       (only a prior: position, direction and size are all measured); average into 1 cm voxels (several frames /
       views merged this way average the depth noise down)
    2. normals: PCA of the 20 nearest neighbours (~2.5 cm patch at 1 cm voxels), oriented towards the camera
    3. RANSAC: two oriented points define a cylinder (axis = n1 x n2, centre = where the two normal lines meet in the
       plane normal to the axis); score = points within 6 mm of the surface (~2.5 sigma of the depth noise
       at 1 m) whose normal agrees (cos > 0.9)
    4. refine: nonlinear least squares on the inliers (axis direction, axis point, radius)
    5. extent: the inliers' range along the axis; the end closer to the camera is the near end
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from world import PipeGeometry


@dataclass
class EstimatedPipe(PipeGeometry):
    end: np.ndarray            # centre of the near end, on the axis
    axis: np.ndarray           # unit, from the near end into the pipe
    radius: float
    length_seen: float
    rms_mm: float
    inliers: int
    points: int
    extra: dict = field(default_factory=dict)

    @property
    def yaw(self):
        return float(np.arctan2(self.axis[1], self.axis[0]))


def voxel_mean(pts, voxel):
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv.ravel(), pts)
    return out / cnt[:, None]


def roi_mask(pts, prior, margin):
    """Points within `margin` of the prior pipe's axis segment (radius included)."""
    a, e = prior.axis, prior.end
    s = np.clip((pts - e) @ a, -margin, prior.length + margin)
    return np.linalg.norm(pts - (e + np.outer(s, a)), axis=1) < prior.radius + margin


def estimate_normals(pts, cam, k=20):
    tree = cKDTree(pts)
    _, nb = tree.query(pts, k=k)
    P = pts[nb] - pts[nb].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", P, P)
    w, V = np.linalg.eigh(C)
    n = V[:, :, 0]                                   # smallest eigenvalue -> normal
    flip = np.sum(n * (cam - pts), 1) < 0
    n[flip] *= -1
    return n


def _axis_dist(p, c, a):
    v = p - c
    return np.linalg.norm(v - np.outer(v @ a, a), axis=1)


def _cyl_from_two(p1, n1, p2, n2, r_range):
    a = np.cross(n1, n2)
    s = np.linalg.norm(a)
    if s < 0.15:
        return None
    a /= s
    # lines p_i + t n_i projected to the plane normal to a; closest point of the two lines = axis point
    m1, m2 = n1 - (n1 @ a) * a, n2 - (n2 @ a) * a
    A = np.stack([m1, -m2], 1)
    t, *_ = np.linalg.lstsq(A, (p2 - (p2 @ a) * a) - (p1 - (p1 @ a) * a), rcond=None)
    c = p1 + t[0] * m1                               # where the two normal lines meet (t[0] < 0: inward)
    r1, r2 = _axis_dist(p1[None], c, a)[0], _axis_dist(p2[None], c, a)[0]
    if abs(r1 - r2) > 0.01 or not (r_range[0] < r1 < r_range[1]):
        return None
    return c, a, 0.5 * (r1 + r2)


def fit_cylinder(pts, cam, prior=None, margin=0.45, iters=500, tol=0.006, r_range=(0.03, 0.5), z_floor=0.03,
                 max_range=2.5, seed=0):
    """pts: world points (any number of frames / views stacked); cam: a camera position (orients the normals).
    prior: a PipeGeometry with .length (e.g. the map's pipe) that defines the region of interest."""
    rng = np.random.default_rng(seed)
    keep = (pts[:, 2] > z_floor) & (np.linalg.norm(pts - cam, axis=1) < max_range)
    if prior is not None:
        keep &= roi_mask(pts, prior, margin)
    p = voxel_mean(pts[keep], 0.01)
    if len(p) < 50:
        raise RuntimeError(f"too few points for a cylinder fit ({len(p)})")
    n = estimate_normals(p, cam)
    tree = cKDTree(p)
    best, best_score = None, 0
    for _ in range(iters):
        i = rng.integers(len(p))
        nb = tree.query_ball_point(p[i], 0.12)      # second point from the same neighbourhood (same object)
        j = rng.choice(nb)
        if j == i:
            continue
        model = _cyl_from_two(p[i], n[i], p[j], n[j], r_range)
        if model is None:
            continue
        c, a, r = model
        dist = _axis_dist(p, c, a)
        radial = (p - c) - np.outer((p - c) @ a, a)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
        inl = (np.abs(dist - r) < tol) & (np.sum(radial * n, 1) > 0.9)
        score = int(inl.sum())
        if score > best_score:
            best, best_score = (c, a, r, inl), score
    if best is None:
        raise RuntimeError("no cylinder found")
    c, a, r, inl = best
    q = p[inl]

    # least-squares refinement: axis = (cos el cos az, cos el sin az, sin el), point c + u e1 + w e2, radius
    def unpack(x):
        az, el, u, w, rr = x
        ax = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        e1 = np.cross(ax, [0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.cross(ax, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        return c + u * e1 + w * e2, ax, rr

    x0 = [np.arctan2(a[1], a[0]), np.arcsin(np.clip(a[2], -1, 1)), 0.0, 0.0, r]
    lo = [-np.inf, -np.inf, -np.inf, -np.inf, r_range[0]]
    hi = [np.inf, np.inf, np.inf, np.inf, r_range[1]]
    x0[4] = float(np.clip(x0[4], r_range[0] + 1e-4, r_range[1] - 1e-4))
    sol = least_squares(lambda x: _axis_dist(q, *unpack(x)[:2]) - unpack(x)[2], x0, loss="soft_l1", f_scale=0.003,
                        bounds=(lo, hi))
    c, a, r = unpack(sol.x)
    res = _axis_dist(q, c, a) - r
    # re-select inliers with the refined model, then the extent along the axis
    dist = _axis_dist(p, c, a)
    inl = np.abs(dist - r) < tol
    s = (p[inl] - c) @ a
    s_lo, s_hi = np.percentile(s, 1), np.percentile(s, 99)
    e_lo, e_hi = c + s_lo * a, c + s_hi * a
    if np.linalg.norm(e_hi - cam) < np.linalg.norm(e_lo - cam):   # point the axis away from the near end
        a, e_lo, s_lo, s_hi = -a, e_hi, -s_hi, -s_lo
    return EstimatedPipe(end=e_lo, axis=a, radius=float(r), length_seen=float(s_hi - s_lo),
                         rms_mm=float(1000 * np.sqrt(np.mean(res ** 2))), inliers=int(inl.sum()), points=len(p),
                         extra=dict(inlier_points=p[inl], cloud=p))


def check_estimate(est, prior, max_shift=0.3, max_tilt_deg=15.0, r_scale=(0.5, 2.0)):
    """Sanity checks against the map prior before acting on a fit. Raises RuntimeError if implausible."""
    v = est.end - prior.end
    shift = np.linalg.norm(v - (v @ prior.axis) * prior.axis)
    tilt = np.degrees(np.arcsin(min(1.0, abs(est.axis[2]))))
    ang = np.degrees(np.arccos(min(1.0, abs(est.axis @ prior.axis))))
    bad = []
    if shift > max_shift:
        bad.append(f"axis {shift:.2f} m from the map's")
    if tilt > max_tilt_deg or ang > 45:
        bad.append(f"axis direction off (tilt {tilt:.0f} deg, {ang:.0f} deg from the map)")
    if not (r_scale[0] * prior.radius < est.radius < r_scale[1] * prior.radius):
        bad.append(f"radius {est.radius:.3f} m implausible")
    if est.length_seen < 0.3:
        bad.append(f"only {est.length_seen:.2f} m of pipe seen")
    if bad:
        raise RuntimeError("cylinder fit rejected: " + "; ".join(bad))
    return est
