"""Measure the work area next to a detected marker - one generic surface model for every vertical work surface.

The marker says only "work here": a rectangle in its own frame (x right, y up, z out of the plate). The robot does
not know if the surface is a flat wall, a pipe or a wavy panel, and it never fits a shape. Instead:

    1. area frame: from the marker pose (ArUco + PnP). Work surfaces are vertical, so the frame is made exactly
       vertical: ey = world up, ez = the marker normal projected on the horizontal plane, ex = ey x ez.
       The area is [0, W] x [0, H] in (u, v); w = distance along ez (towards the robot) from the marker plane.
    2. depth points (head depth camera looking straight at the surface) -> (u, v, w) coordinates, cropped to the
       area (+ a small margin) and to a +-0.3 m slab around the marker plane
    3. sim/perception/surface_fit.py fit_surface on (u, v, w): per cell the front-most surface (median of the points
       within 2 cm of the largest w), one
       global thin-plate spline, bicubic resample -> w(u, v) with C2 heights and analytic normals.
A flat wall comes out as w = const, a pipe as a half-cylinder bulge, a panel as its waves - same code path.
The model is kept in area coordinates, so it moves with the marker's frame when the robot re-localises on it.
"""
from dataclasses import dataclass

import numpy as np

from perception.surface_fit import fit_surface


@dataclass
class AreaFrame:
    o: np.ndarray        # area lower-left corner, on the marker plane
    ex: np.ndarray       # right (horizontal)
    ey: np.ndarray       # up
    ez: np.ndarray       # out of the surface, towards the robot (horizontal)
    W: float
    H: float

    def to_area(self, P):
        r = np.asarray(P, float) - self.o
        return np.stack([r @ self.ex, r @ self.ey, r @ self.ez], -1)

    def to_world(self, u, v, w):
        return self.o + np.multiply.outer(u, self.ex) + np.multiply.outer(v, self.ey) + np.multiply.outer(w, self.ez)

    def mapped(self, f_point, f_vec):
        """The same frame under a rigid transform (e.g. the robot's map frame -> the physics frame)."""
        return AreaFrame(f_point(self.o), f_vec(self.ex), f_vec(self.ey), f_vec(self.ez), self.W, self.H)


def area_frame(marker_pos, marker_R, offset, size):
    """Vertical area frame from a marker pose and a work order (centre offset + size in the marker frame)."""
    z = np.array([*marker_R[:2, 2], 0.0])
    z /= np.linalg.norm(z)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    W, H = size
    o = np.asarray(marker_pos, float) + (offset[0] - W / 2) * x + (offset[1] - H / 2) * y
    return AreaFrame(o, x, y, z, W, H)


class MeasuredSurface:
    """w(u, v) over the work area, in area coordinates (independent of where the frame is in the world)."""

    def __init__(self, model, W, H, info):
        self.model, self.W, self.H, self.info = model, W, H, info

    def w(self, u, v):
        return self.model.height(u, v)

    def normal_uvw(self, u, v):
        return self.model.normal(u, v)          # (-w_u, -w_v, 1) normalised: points out of the surface

    def covered(self, u, v):
        return self.model.covered(u, v)


def fit_area_surface(pts, frame: AreaFrame, edge=0.015, slab=0.3, min_coverage=0.9, band=0.02):
    """pts: depth points (same world frame as `frame`). edge: model margin beyond the area (the tip and the stroke
    ends need a little surface outside the rectangle). band: per cell, points within `band` of the front-most one
    are averaged (median) - wide enough (~5 sigma of depth noise at 1-1.3 m) that the front-most noisy point does
    not bias the surface towards the camera, far tighter than any background step (pipe -> wall: >= 10 cm)."""
    q = frame.to_area(pts)
    q = q[np.abs(q[:, 2]) < slab]
    W, H = frame.W, frame.H
    model = fit_surface(q, roi=(W / 2, H / 2, W / 2 + edge, H / 2 + edge), margin=0.0, band=band)
    U, V = np.meshgrid(np.linspace(0, W, 25), np.linspace(0, H, 9))
    cov = float(model.covered(U, V).mean())
    if cov < min_coverage:
        raise RuntimeError(f"work area only {100 * cov:.0f} % observed")
    w = model.height(U, V)
    info = dict(model.info, area_coverage=cov, depth_range_mm=[float(1000 * w.min()), float(1000 * w.max())])
    return MeasuredSurface(model, W, H, info)
