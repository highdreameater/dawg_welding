"""Measure the work surface next to a detected marker from head depth-camera point clouds (no map knowledge of it).

    fit_wall_plane : points around the expected paint area -> RANSAC plane (normal near horizontal) -> SVD refine
    fit_pipe_near  : points within the work order's search radius of the sign, floor / sign removed ->
                     RANSAC + least-squares cylinder (sim/perception/cylinder_fit.py, no prior shape or size) ->
                     axis point at the work height, radius, axis tilt
"""
from dataclasses import dataclass

import numpy as np

from perception.cylinder_fit import fit_cylinder, voxel_mean


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray      # unit, pointing towards the camera
    rms_mm: float
    inliers: int


def fit_wall_plane(pts, cam, center, half=(0.6, 0.35), depth=0.4, iters=300, tol=0.008, seed=0):
    """center: rough area centre (from the marker). Keeps points within `half` (lateral, vertical) and `depth` of it."""
    rng = np.random.default_rng(seed)
    v = pts - center
    keep = (np.abs(v[:, 2]) < half[1]) & (np.linalg.norm(v[:, :2], axis=1) < np.hypot(half[0], depth))
    p = voxel_mean(pts[keep], 0.01)
    if len(p) < 50:
        raise RuntimeError(f"too few wall points ({len(p)})")
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) > 0.3:          # a wall: near-vertical plane
            continue
        inl = np.abs((p - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        raise RuntimeError("no wall plane found")
    q = p[best]
    c0 = q.mean(0)
    _, _, Vt = np.linalg.svd(q - c0)
    n = Vt[2]
    if n @ (cam - c0) < 0:
        n = -n
    res = (q - c0) @ n
    return Plane(c0, n, float(1000 * np.sqrt(np.mean(res ** 2))), int(len(q)))


@dataclass
class VPipe:
    xy: np.ndarray          # axis point (x, y) at the work height
    radius: float
    tilt_deg: float
    rms_mm: float
    inliers: int
    length_seen: float
    points: np.ndarray      # inlier points (for drawing)


def fit_pipe_near(pts, cam, sign_pos, sign_normal, search_radius, ground_z, z_work, remove_sign=True):
    """Cylinder within `search_radius` of the sign. The sign board itself and the floor are cut out first."""
    d = pts[:, :2] - sign_pos[:2]
    keep = np.linalg.norm(d, axis=1) < search_radius
    keep &= pts[:, 2] > ground_z + 0.10
    keep &= pts[:, 2] < ground_z + 1.5
    # sign board + post: within 6 cm of the board plane and 0.25 m of its centre line
    n = np.array([*sign_normal[:2], 0.0]); n /= np.linalg.norm(n)
    rel = pts - sign_pos
    board = (np.abs(rel @ n) < 0.08) & (np.linalg.norm(rel[:, :2] - np.outer(rel @ n, n)[:, :2], axis=1) < 0.25)
    if remove_sign:
        keep &= ~board
    est = fit_cylinder(pts[keep], cam, prior=None, r_range=(0.04, 0.3), max_range=3.0)
    a = est.axis if est.axis[2] > 0 else -est.axis
    tilt = float(np.degrees(np.arccos(np.clip(a[2], -1, 1))))
    if tilt > 10:
        raise RuntimeError(f"cylinder near the sign is not vertical (tilt {tilt:.0f} deg)")
    s = (z_work - est.end[2]) / a[2]
    xy = (est.end + s * a)[:2]
    return VPipe(xy, est.radius, tilt, est.rms_mm, est.inliers, est.length_seen, est.extra["inlier_points"])
