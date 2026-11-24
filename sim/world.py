"""Welding workshop world: walls, obstacles and work stations, added to a robot MjSpec, plus a 2-D occupancy map.

The world is fully known (no SLAM): the path planner reads the same obstacle list that builds the MuJoCo scene.

    room           10.5 m x 8 m, robot starts at (0, 0) facing +x
    obstacles      crate stacks, a steel column, a shelving rack, gas bottles
    station "pipe" horizontal pipe (r = 0.15 m) on two saddles, axis along +x: curved surface for force control
    station "beam" the flat beam with a straight seam used by sim/stance_demo.py

Geom groups / collision bits:
    world geoms: group 0 (seen by the locomotion height scan), contype = conaffinity = 1
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


@dataclass
class Pipe:                # horizontal, axis along +x from x0 to x0 + length
    name: str
    x0: float
    y: float
    z: float               # axis height
    length: float
    radius: float
    rgba: list = field(default_factory=lambda: STEEL)

    @property
    def axis(self):
        return np.array([1.0, 0.0, 0.0])

    def surface_point(self, x, theta, offset=0.0):
        """Point on the pipe surface (plus `offset` along the normal) at axial position x and angle theta
        (theta = 0: top, > 0: towards +y). Returns (point, outward normal)."""
        n = np.array([0.0, np.sin(theta), np.cos(theta)])
        return np.array([x, self.y, self.z]) + (self.radius + offset) * n, n


@dataclass
class Station:
    name: str
    base_xy: tuple         # where the base centre should stand
    base_yaw: float
    kind: str              # "pipe" (force control) or "beam" (stance weld)
    work: object = None    # the workpiece geometry


@dataclass
class World:
    bounds: tuple                       # (xmin, xmax, ymin, ymax) inner wall faces
    boxes: list
    cylinders: list
    pipes: list
    stations: dict
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
            g.rgba = b.rgba
        for c in self.cylinders:
            g = wb.add_geom()
            g.name, g.type = c.name, mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size, g.pos, g.rgba = [c.radius, c.height / 2, 0], [*c.xy, c.height / 2], c.rgba
        for p in self.pipes:
            g = wb.add_geom()
            g.name, g.type = p.name, mujoco.mjtGeom.mjGEOM_CYLINDER
            g.fromto = [p.x0, p.y, p.z, p.x0 + p.length, p.y, p.z]
            g.size, g.rgba = [p.radius, 0, 0], p.rgba
            g.contype = g.conaffinity = 3
            g.solref = [0.005, 1.0]  # steel: stiffer than the default contact
            for k, xs in enumerate((p.x0 + 0.25, p.x0 + p.length - 0.25)):  # saddles
                s = wb.add_geom()
                s.name, s.type, s.rgba = f"{p.name}_saddle{k}", mujoco.mjtGeom.mjGEOM_BOX, PAINT
                hz = (p.z - 0.6 * p.radius) / 2
                s.size, s.pos = [0.05, 0.8 * p.radius, hz], [xs, p.y, hz]
        light = wb.add_light()
        light.pos, light.dir = [(x0 + x1) / 2, (y0 + y1) / 2, 6.0], [0, 0, -1]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [0.35, 0.35, 0.35]

    # ------------------------------------------------------------------ 2-D map
    def footprints(self):
        """Obstacle footprints for the planner: ('box', cx, cy, hx, hy, yaw) / ('circle', cx, cy, r)."""
        out = [("box", b.center[0], b.center[1], b.half[0], b.half[1], b.yaw) for b in self.boxes]
        out += [("circle", c.xy[0], c.xy[1], c.radius) for c in self.cylinders]
        out += [("box", p.x0 + p.length / 2, p.y, p.length / 2, p.radius, 0.0) for p in self.pipes]
        return out


def workshop():
    pipe = Pipe("pipe", x0=7.30, y=-2.30, z=0.17, length=1.60, radius=0.15)
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
    stations = {
        # work on the pipe top 8 cm in from its end; base centre 0.56 m behind that, facing +x (nose overhangs
        # the low pipe end by ~4 cm, ~7 cm above it)
        "pipe": Station("pipe", (pipe.x0 + PIPE_WORK_X - PIPE_WORK_DIST, pipe.y), 0.0, "pipe", pipe),
        # same relative placement as sim/stance_demo.py (seam 0.70 m ahead of the base)
        "beam": Station("beam", (beam.center[0] - 0.70, beam.center[1]), 0.0, "beam", beam),
    }
    return World(bounds=(-1.5, 9.0, -4.0, 4.0), boxes=boxes, cylinders=cylinders, pipes=[pipe], stations=stations)


WORLDS = {"workshop": workshop}
