"""Welding workshop world: walls, obstacles and work stations, added to a robot MjSpec, plus a 2-D occupancy map.

The world is fully known (no SLAM): the path planner reads the same obstacle list that builds the MuJoCo scene.

    room           10.5 m x 8 m, robot starts at (0, 0) facing +x
    obstacles      crate stacks, a steel column, a shelving rack, gas bottles
    station "pipe" horizontal pipe (r = 0.15 m) on two saddles, axis along +x: curved surface for force control
    station "beam" the flat beam with a straight seam used by sim/stance_demo.py

Geom groups / collision bits:
    floor:       group 0 - the only thing the locomotion height scan sees (terrain)
    world geoms: group 1 - obstacles / workpieces (avoided by the planner, seen by the depth camera),
                 contype = conaffinity = 1
    pipe:        contype = conaffinity = 3 -> also touched by the contact tip (bit 2, see robots.py)
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

PIPE_WORK_X = 0.08  # force-control arc sits this far in from the pipe end [m]
PIPE_WORK_DIST = 0.56  # base centre stands this far behind the arc (Z1 reach with a tilted torch, see README) [m]
STEEL, CRATE, RACK, PAINT = [0.45, 0.47, 0.5, 1], [0.62, 0.45, 0.25, 1], [0.2, 0.3, 0.55, 1], [0.85, 0.65, 0.1, 1]


@dataclass
class Box:
    name: str
    center: tuple          # (x, y, z) of the box centre
    half: tuple            # half sizes
    yaw: float = 0.0
    rgba: list = field(default_factory=lambda: CRATE)


@dataclass
class Cylinder:            # vertical
    name: str
    xy: tuple
    radius: float
    height: float
    rgba: list = field(default_factory=lambda: STEEL)


def pipe_frame(axis):
    """Unit vectors (up, side) perpendicular to a pipe axis: up = closest to world +z, side = up x axis."""
    a = np.asarray(axis, float)
    up = np.array([0.0, 0.0, 1.0]) - a[2] * a
    up /= np.linalg.norm(up)
    return up, np.cross(up, a)


class PipeGeometry:
    """Cylinder interface used by the force task. s = axial distance from the near end, theta = angle around the
    axis (0 = top, > 0 towards `side`). Subclasses: Pipe (the world / map) and EstimatedPipe (perception)."""
    end: np.ndarray            # centre of the near end (on the axis)
    axis: np.ndarray           # unit, pointing from the near end into the pipe
    radius: float

    def normal(self, theta):
        up, side = pipe_frame(self.axis)
        return np.cos(theta) * up + np.sin(theta) * side

    def surface_point(self, s, theta, offset=0.0):
        """Point on the surface (plus `offset` along the normal) and the outward normal."""
        n = self.normal(theta)
        return self.end + s * self.axis + (self.radius + offset) * n, n


@dataclass
class Pipe(PipeGeometry):  # horizontal, near end centre at (x0, y, z), axis at heading `yaw`
    name: str
    x0: float
    y: float
    z: float               # axis height
    length: float
    radius: float
    yaw: float = 0.0
    rgba: list = field(default_factory=lambda: STEEL)

    @property
    def axis(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    @property
    def end(self):
        return np.array([self.x0, self.y, self.z])


@dataclass
class FreeformPanel:
    """A random smooth surface (no closed-form shape) on a stand: z = top_z + f(x - cx, y - cy) over a
    2 hx x 2 hy rectangle. f = random Gaussian bumps + a random wave, rescaled to +-amp; built in MuJoCo as a
    heightfield (`res` grid). f and its gradient are evaluated analytically, which is the ground truth for scoring."""
    name: str
    cx: float
    cy: float
    hx: float
    hy: float
    top_z: float                # mean surface height
    amp: float = 0.03           # max |height deviation|
    seed: int = 0
    res: float = 0.004          # heightfield grid [m]
    rgba: list = field(default_factory=lambda: [0.55, 0.57, 0.6, 1])

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        k = 7
        self._b = np.stack([rng.uniform(-self.hx, self.hx, k), rng.uniform(-self.hy, self.hy, k),
                            rng.uniform(0.05, 0.10, k), rng.uniform(-1, 1, k)], 1)        # u0, v0, sigma, a
        ang, lam = rng.uniform(0, np.pi), rng.uniform(0.25, 0.4)
        self._w = (np.cos(ang) * 2 * np.pi / lam, np.sin(ang) * 2 * np.pi / lam, rng.uniform(0, 2 * np.pi), 0.35)
        u, v = np.meshgrid(np.linspace(-self.hx, self.hx, 81), np.linspace(-self.hy, self.hy, 81))
        raw = self._raw(u, v)
        self._mean = raw.mean()
        self._scale = self.amp / np.abs(raw - self._mean).max()

    def _raw(self, u, v, grad=False):
        f, fu, fv = 0.0, 0.0, 0.0
        for u0, v0, sg, a in self._b:
            g = a * np.exp(-((u - u0) ** 2 + (v - v0) ** 2) / (2 * sg ** 2))
            f = f + g
            fu, fv = fu - g * (u - u0) / sg ** 2, fv - g * (v - v0) / sg ** 2
        ku, kv, ph, a = self._w
        f = f + a * np.sin(ku * u + kv * v + ph)
        c = a * np.cos(ku * u + kv * v + ph)
        return (f, fu + c * ku, fv + c * kv) if grad else f

    def height(self, x, y):
        return self.top_z + self._scale * (self._raw(x - self.cx, y - self.cy) - self._mean)

    def normal(self, x, y):
        _, fu, fv = self._raw(x - self.cx, y - self.cy, grad=True)
        n = np.stack(np.broadcast_arrays(-self._scale * fu, -self._scale * fv, np.ones_like(np.asarray(x, float))), -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def add_to(self, spec, wb):
        ncol, nrow = int(round(2 * self.hx / self.res)) + 1, int(round(2 * self.hy / self.res)) + 1
        X, Y = np.meshgrid(np.linspace(self.cx - self.hx, self.cx + self.hx, ncol),
                           np.linspace(self.cy - self.hy, self.cy + self.hy, nrow))   # row index = y (MuJoCo order)
        H = self.height(X, Y)
        zlo, zhi = H.min(), H.max()             # MuJoCo re-normalises the elevation data to [0, 1]
        hf = spec.add_hfield()
        hf.name, hf.nrow, hf.ncol = self.name, nrow, ncol
        hf.size = [self.hx, self.hy, zhi - zlo, 0.02]
        hf.userdata = ((H - zlo) / (zhi - zlo)).ravel().tolist()
        g = wb.add_geom()
        g.name, g.type, g.hfieldname = self.name, mujoco.mjtGeom.mjGEOM_HFIELD, self.name
        g.pos, g.rgba, g.group = [self.cx, self.cy, zlo], self.rgba, 1
        g.contype = g.conaffinity = 3           # also touched by the contact tip (bit 2)
        g.solref = [0.005, 1.0]
        st = wb.add_geom()                      # stand under the slab
        st.name, st.type, st.rgba, st.group = f"{self.name}_stand", mujoco.mjtGeom.mjGEOM_BOX, PAINT, 1
        h = (zlo - 0.02) / 2
        st.size, st.pos = [self.hx - 0.01, self.hy - 0.01, h], [self.cx, self.cy, h]


@dataclass
class Station:
    name: str
    base_xy: tuple         # where the base centre should stand (pipe: for ANYmal-C; see pipe_dock_pose)
    base_yaw: float
    kind: str              # "pipe" (force control) or "beam" (stance weld)
    work: object = None    # the workpiece geometry *as in the map* (the real one may differ, see World.pipes)


def pipe_dock_pose(pipe: PipeGeometry, work_dist):
    """Base pose (x, y, yaw) facing along the pipe axis, `work_dist` behind the work arc (PIPE_WORK_X in)."""
    a = np.array([pipe.axis[0], pipe.axis[1]])
    a /= np.linalg.norm(a)
    xy = pipe.end[:2] + (PIPE_WORK_X - work_dist) * a
    return np.array([xy[0], xy[1], np.arctan2(a[1], a[0])])


def pipe_inspect_pose(pipe: PipeGeometry, work_dist, back=0.5, side=0.6):
    """Where to stand to look at the pipe with a head camera: `back` behind and `side` to the left of the docking
    pose, facing a point 0.35 m into the pipe. Head-on views see the curved surface only at grazing angles (the
    Go2 camera is ~4 cm above the pipe top) and the cylinder fit fails; from the side it sees ~half the pipe."""
    dock = pipe_dock_pose(pipe, work_dist)
    a = np.array([np.cos(dock[2]), np.sin(dock[2])])
    xy = dock[:2] - back * a + side * np.array([-a[1], a[0]])
    look = pipe.end[:2] + 0.35 * a
    return np.array([xy[0], xy[1], np.arctan2(look[1] - xy[1], look[0] - xy[0])])


@dataclass
class World:
    bounds: tuple                       # (xmin, xmax, ymin, ymax) inner wall faces
    boxes: list
    cylinders: list
    pipes: list                         # the real pipes (MuJoCo scene)
    stations: dict
    map_pipes: list = None              # the pipes as the map knows them (planner); default = the real ones
    panels: list = field(default_factory=list)   # freeform surfaces (the map knows only their stand footprint)
    start: tuple = (0.0, 0.0, 0.0)      # robot start (x, y, yaw)
    wall_h: float = 1.2

    # ------------------------------------------------------------------ MuJoCo
    def add_to(self, spec: mujoco.MjSpec):
        wb = spec.worldbody
        x0, x1, y0, y1 = self.bounds
        t = 0.1
        walls = [Box("wall_s", ((x0 + x1) / 2, y0 - t, self.wall_h / 2), ((x1 - x0) / 2 + 2 * t, t, self.wall_h / 2)),
                 Box("wall_n", ((x0 + x1) / 2, y1 + t, self.wall_h / 2), ((x1 - x0) / 2 + 2 * t, t, self.wall_h / 2)),
                 Box("wall_w", (x0 - t, (y0 + y1) / 2, self.wall_h / 2), (t, (y1 - y0) / 2, self.wall_h / 2)),
                 Box("wall_e", (x1 + t, (y0 + y1) / 2, self.wall_h / 2), (t, (y1 - y0) / 2, self.wall_h / 2))]
        for b in walls:
            b.rgba = [0.7, 0.72, 0.75, 1]
        for b in walls + self.boxes:
            g = wb.add_geom()
            g.name, g.type, g.size, g.pos = b.name, mujoco.mjtGeom.mjGEOM_BOX, list(b.half), list(b.center)
            g.quat = [np.cos(b.yaw / 2), 0, 0, np.sin(b.yaw / 2)]
            g.rgba, g.group = b.rgba, 1
        for c in self.cylinders:
            g = wb.add_geom()
            g.name, g.type = c.name, mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size, g.pos, g.rgba, g.group = [c.radius, c.height / 2, 0], [*c.xy, c.height / 2], c.rgba, 1
        for p in self.pipes:
            g = wb.add_geom()
            g.name, g.type = p.name, mujoco.mjtGeom.mjGEOM_CYLINDER
            g.fromto = [*p.end, *(p.end + p.length * p.axis)]
            g.size, g.rgba, g.group = [p.radius, 0, 0], p.rgba, 1
            g.contype = g.conaffinity = 3
            g.solref = [0.005, 1.0]  # steel: stiffer than the default contact
            for k, sk in enumerate((0.25, p.length - 0.25)):  # saddles
                s = wb.add_geom()
                s.name, s.type, s.rgba = f"{p.name}_saddle{k}", mujoco.mjtGeom.mjGEOM_BOX, PAINT
                hz = (p.z - 0.6 * p.radius) / 2
                c = p.end + sk * p.axis
                s.size, s.pos, s.group = [0.05, 0.8 * p.radius, hz], [c[0], c[1], hz], 1
                s.quat = [np.cos(p.yaw / 2), 0, 0, np.sin(p.yaw / 2)]
        for pn in self.panels:
            pn.add_to(spec, wb)
        light = wb.add_light()
        light.pos, light.dir = [(x0 + x1) / 2, (y0 + y1) / 2, 6.0], [0, 0, -1]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [0.35, 0.35, 0.35]

    # ------------------------------------------------------------------ 2-D map
    def footprints(self):
        """Obstacle footprints for the planner: ('box', cx, cy, hx, hy, yaw) / ('circle', cx, cy, r)."""
        out = [("box", b.center[0], b.center[1], b.half[0], b.half[1], b.yaw) for b in self.boxes]
        out += [("circle", c.xy[0], c.xy[1], c.radius) for c in self.cylinders]
        for p in (self.map_pipes or self.pipes):
            c = p.end + 0.5 * p.length * p.axis
            out.append(("box", c[0], c[1], p.length / 2, p.radius, p.yaw))
        out += [("box", pn.cx, pn.cy, pn.hx, pn.hy, 0.0) for pn in self.panels]
        return out


def panel_dock_pose(panel: FreeformPanel, work_dist):
    """Base pose facing +x with the panel's near edge PIPE_WORK_X behind the work point `work_dist` ahead."""
    return np.array([panel.cx - panel.hx + PIPE_WORK_X - work_dist, panel.cy, 0.0])


def workshop(pipe_dx=0.0, pipe_dy=0.0, pipe_dyaw_deg=0.0, pipe_radius=None, panel_seed=0, panel_amp=0.03):
    """pipe_*: move / turn / resize the real pipe relative to the map (the robot only knows the map + its sensors).
    panel_seed / panel_amp: the random freeform surface at the "panel" station (the map knows only its stand)."""
    pipe = Pipe("pipe", x0=7.30, y=-2.30, z=0.17, length=1.60, radius=0.15)   # as in the map
    r = pipe.radius if pipe_radius is None else pipe_radius
    real = Pipe("pipe", x0=pipe.x0 + pipe_dx, y=pipe.y + pipe_dy, z=r + 0.02, length=pipe.length, radius=r,
                yaw=np.radians(pipe_dyaw_deg))
    beam = Box("beam", (7.40, 2.30, 0.15), (0.08, 0.30, 0.15), rgba=STEEL)
    boxes = [
        Box("crates_a", (2.40, 0.50, 0.45), (0.40, 0.70, 0.45)),
        Box("crates_b", (2.80, -2.80, 0.35), (0.50, 0.40, 0.35)),
        Box("crate_c", (1.20, 2.60, 0.30), (0.35, 0.35, 0.30), yaw=0.4),
        Box("rack", (4.30, 1.90, 0.90), (0.90, 0.30, 0.90), rgba=RACK),
        Box("table", (5.90, -0.30, 0.40), (0.60, 0.40, 0.40), rgba=[0.35, 0.35, 0.38, 1]),
        beam,
    ]
    cylinders = [
        Cylinder("column", (4.30, -1.30), 0.25, 1.2),
        Cylinder("gas_1", (7.00, 1.00), 0.12, 0.9, rgba=[0.15, 0.5, 0.25, 1]),
        Cylinder("gas_2", (7.00, 1.30), 0.12, 0.9, rgba=[0.15, 0.5, 0.25, 1]),
    ]
    panel = FreeformPanel("panel", cx=8.33, cy=-0.60, hx=0.13, hy=0.25, top_z=0.31, amp=panel_amp, seed=panel_seed)
    stations = {
        # random freeform surface on a stand, robot docks facing +x at its near edge
        "panel": Station("panel", tuple(panel_dock_pose(panel, PIPE_WORK_DIST)[:2]), 0.0, "panel", panel),
        # work on the pipe top 8 cm in from its end; base centre 0.56 m behind that, facing +x (nose overhangs
        # the low pipe end by ~4 cm, ~7 cm above it)
        "pipe": Station("pipe", tuple(pipe_dock_pose(pipe, PIPE_WORK_DIST)[:2]), 0.0, "pipe", pipe),
        # same relative placement as sim/stance_demo.py (seam 0.70 m ahead of the base)
        "beam": Station("beam", (beam.center[0] - 0.70, beam.center[1]), 0.0, "beam", beam),
    }
    return World(bounds=(-1.5, 9.0, -4.0, 4.0), boxes=boxes, cylinders=cylinders, pipes=[real], stations=stations,
                 map_pipes=[pipe], panels=[panel])


WORLDS = {"workshop": workshop}
