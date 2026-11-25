"""Marker targets: a registry of detected / visited ArUco markers, and approach poses in front of a marker.

    reg = MarkerRegistry(n_expected=4)
    new_ids = reg.update(observations)           # MarkerObservation list -> fused estimates; ids newly confirmed
    tgt = reg.next_target(robot_xy)              # nearest detected, not yet visited marker (None if there is none)
    reg.mark_visited(tgt.id, t, alignment)       # never selected again
    path, info, pose = plan_to_marker(grid, robot_xy, tgt.p, tgt.n)   # reuses grid_planner.plan_path

Fusion (per marker id; ids are unique, so there is no data association): weighted means of the PnP positions and
normals. Weights favour close, frontal views (1 / d^2 * cos(view)). The normal of a near-frontal view is ambiguous
for planar PnP (IPPE returns two solutions with similar error, see perception/aruco.py), so a view's normal weight is
also scaled by (1 - ambiguity)^2. After 3 views, observations more than max(0.3 m, 0.15 d) from the estimate are
rejected as outliers.
"""
from dataclasses import dataclass, field

import numpy as np

from planning.grid_planner import plan_path


@dataclass
class MarkerTrack:
    id: int
    p: np.ndarray
    n: np.ndarray
    first_t: float
    last_t: float
    n_obs: int = 0
    best_dist: float = np.inf
    status: str = "detected"          # candidate -> detected | rejected;  detected -> visited | unreachable
    visit: dict = None
    _wp: float = 0.0
    _wn: float = 0.0
    _P: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rejected: int = 0

    def add(self, o):
        w = max(np.cos(np.radians(o.view_deg)), 0.2) / max(o.dist, 0.3) ** 2
        if self.n_obs >= 3 and np.linalg.norm(o.p - self.p) > max(0.3, 0.15 * o.dist):
            self.rejected += 1
            return False
        wn = w * (1.0 - min(o.ambiguity, 1.0)) ** 2 + 1e-6 * w
        self._wp += w
        self._P += w * o.p
        self._wn += wn
        self._N += wn * o.n
        self.p = self._P / self._wp
        self.n = self._N / np.linalg.norm(self._N)
        self.n_obs += 1
        self.last_t = o.t
        self.best_dist = min(self.best_dist, o.dist)
        return True


class MarkerRegistry:
    """A new id is a *candidate* until it has `min_views` consistent observations and (if given) `plausible(p)` holds
    for its fused position - e.g. "lies on a surface of the map". Only then is it *detected* (a target). This rejects
    the occasional single-frame false positive of the small 4x4 dictionary (seen: a phantom id 17)."""

    def __init__(self, n_expected=4, min_views=3, plausible=None):
        self.n_expected, self.min_views, self.plausible = n_expected, min_views, plausible
        self.tracks = {}
        self.order_detected, self.order_visited = [], []

    def update(self, observations):
        """Fuse observations; returns the ids that became detected (confirmed) with this update."""
        new = []
        for o in observations:
            tr = self.tracks.get(o.id)
            if tr is None:
                tr = self.tracks[o.id] = MarkerTrack(o.id, o.p.copy(), o.n.copy(), o.t, o.t, status="candidate")
            tr.add(o)
            if (tr.status == "candidate" and tr.n_obs >= self.min_views
                    and (self.plausible is None or self.plausible(tr.p))):
                tr.status = "detected"
                self.order_detected.append(o.id)
                new.append(o.id)
        return new

    def candidates(self):
        """Ids seen but never confirmed (false positives, or markers glimpsed too briefly) -> number of views."""
        return {i: t.n_obs for i, t in self.tracks.items() if t.status == "candidate"}

    def next_candidate(self, robot_xy):
        """Nearest unconfirmed candidate whose estimate is plausible (worth a closer look), or None."""
        c = [t for t in self.tracks.values()
             if t.status == "candidate" and (self.plausible is None or self.plausible(t.p))]
        return min(c, key=lambda t: np.linalg.norm(t.p[:2] - np.asarray(robot_xy))) if c else None

    def reject(self, mid):
        """A candidate that was not seen again from close up: a false positive."""
        self.tracks[mid].status = "rejected"

    def unvisited(self):
        return [t for t in self.tracks.values() if t.status == "detected"]

    def next_target(self, robot_xy):
        """Nearest (straight line) detected marker that has not been visited and is not known to be unreachable."""
        cand = self.unvisited()
        if not cand:
            return None
        return min(cand, key=lambda t: np.linalg.norm(t.p[:2] - np.asarray(robot_xy)))

    def mark_visited(self, mid, t, info=None):
        tr = self.tracks[mid]
        if tr.status == "visited":
            raise RuntimeError(f"marker {mid} visited twice")
        tr.status, tr.visit = "visited", dict(t=t, **(info or {}))
        self.order_visited.append(mid)

    def mark_unreachable(self, mid, reason):
        self.tracks[mid].status, self.tracks[mid].visit = "unreachable", dict(reason=reason)

    @property
    def n_visited(self):
        return len(self.order_visited)

    @property
    def complete(self):
        return self.n_visited >= self.n_expected


def approach_poses(p, n, robot_xy=None, standoffs=(0.75, 0.9), yaw_offsets_deg=(0, 15, -15, 30, -30)):
    """Candidate base poses (x, y, yaw) in front of a marker, facing it: `standoff` from the marker centre along its
    horizontal normal (rotated by the yaw offset), best first. A marker facing (almost) straight up has no horizontal
    normal; then the robot stands on the side it comes from."""
    nh = np.asarray(n[:2], float)
    if np.linalg.norm(nh) < 0.3:
        nh = np.asarray(robot_xy, float) - p[:2] if robot_xy is not None else np.array([1.0, 0.0])
    nh = nh / np.linalg.norm(nh)
    out = []
    for off in yaw_offsets_deg:
        for d in standoffs:
            a = np.radians(off)
            u = np.array([np.cos(a) * nh[0] - np.sin(a) * nh[1], np.sin(a) * nh[0] + np.cos(a) * nh[1]])
            xy = p[:2] + d * u
            out.append((np.array([xy[0], xy[1], np.arctan2(-u[1], -u[0])]), d, off))
    return out


def plan_to_marker(grid, start_xy, p, n, approaches=(0.8, 0.5, 0.25), **kw):
    """Plan to the best feasible approach pose: the pose must be free for the robot disc (it turns in place there),
    the straight docking leg (`approach` m, longest that fits) must be clear. Returns (path, info, pose, standoff,
    yaw offset). Raises RuntimeError if no pose is reachable."""
    errors = []
    for pose, d, off in approach_poses(p, n, start_xy, **kw):
        if not grid.free[grid.to_cell(pose[:2])]:
            errors.append(f"({off} deg, {d} m): pose in collision")
            continue
        for ap in approaches:
            try:
                path, info = plan_path(grid, start_xy, pose[:2], pose[2], approach=ap)
                info["approach_m"] = ap
                return path, info, pose, d, off
            except RuntimeError as e:
                err = str(e)
        errors.append(f"({off} deg, {d} m): {err}")
    raise RuntimeError("no reachable approach pose: " + "; ".join(errors))
