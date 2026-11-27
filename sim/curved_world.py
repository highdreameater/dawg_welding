"""Curved-surface ArUco world: the user's terrain corridor (sim/aruco_world.py: displaced floor, walls, full-width
staircase, crates / barrels, a column) with four non-flat work surfaces, one ArUco marker on each.

    vessel       upright ellipsoid (0.40 x 0.40 x 0.60 m) at the north wall        marker  5, faces south
    pipe         horizontal pipe (r 0.16 m, axis 0.66 m up) on two saddles, south    marker 23, on its north flank,
                 wall                                                               10 deg below the axis
    sphere_tank  spherical tank (r 0.45 m) on a pedestal, south wall, far end       marker 37, faces north
    panel        the random wavy panel of mission v8 (FreeformPanel heightfield,     marker  7, a flat sticker on the
                 +-3 cm, slopes up to ~30 deg) on the south wall, beyond the stairs  wall just below the panel

The markers are printed sheets stuck on the surface: they bend with it (sim/markers.py - unrolled onto cylinders,
projected onto ellipsoids, flat on the wall under the panel). Surface + marker classes are the ones of sim/world.py (Pipe, Ellipsoid,
ArucoMarker), so the geometry is the same as in sim/aruco_mission.py.

What the robot knows (the map): walls, staircase, every obstacle - including the four objects as obstacles (it knows a
tank stands there). What it does not know: which of them carry a marker, where the marker is, the exact surface.

Work orders (marker id -> rectangle to cover, in the marker's vertical frame: x right, y up; centre offset + size).
Work bands: 0.57-0.75 m above the local ground on the vessel, tank and panel (the area above the marker), 0.61-0.71 m
on the horizontal pipe (beside the marker, around the axis height). The markers are at 0.40-0.63 m, about the height of the
Go2's head camera, so they are in view from the pre-dock pose. The pipe sits at the height where the arm reaches its
whole strip from a range of standoffs (0.64-0.70 m); at 0.55 m only one standoff (0.79 m) reached it and any dock error
cost coverage.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from aruco_world import ArucoWorld, Box, Stairs, Terrain, VerticalPanel
from world import ArucoMarker, Ellipsoid, Pipe

MARKER_SIZE = 0.15          # black square edge of the printed markers [m]
STEEL = [0.45, 0.47, 0.5, 1]

WORK_ORDERS = {
    5: dict(area_offset=(0.0, 0.26), area_size=(0.30, 0.16), color=(0.10, 0.55, 0.20)),     # vessel
    23: dict(area_offset=(-0.33, 0.03), area_size=(0.30, 0.10), color=(0.85, 0.12, 0.10)),  # pipe: beside the marker
    37: dict(area_offset=(0.0, 0.26), area_size=(0.24, 0.14), color=(0.95, 0.55, 0.05)),    # sphere tank
    7: dict(area_offset=(0.0, 0.26), area_size=(0.40, 0.16), color=(0.55, 0.15, 0.75)),     # wavy panel (as in v8)
}


@dataclass
class WallPlane:
    """A flat wall face as a 'surface' for sim/markers.py (a sticker on it is flat): point on the face + outward normal."""
    name: str
    point: tuple
    normal: tuple

    @property
    def center(self):
        return np.array([self.point[0], self.point[1], 0.0])

    def normal_at(self, p):
        n = np.asarray(self.normal, float)
        return n / np.linalg.norm(n)

    def ray(self, o, d):
        n = self.normal_at(None)
        den = float(np.asarray(d, float) @ n)
        if abs(den) < 1e-9:
            return None
        t = float((np.asarray(self.point, float) - np.asarray(o, float)) @ n) / den
        return t if t > 1e-9 else None


@dataclass
class CurvedWorld(ArucoWorld):
    cyl_surfaces: list = field(default_factory=list)       # Cylinder (vertical)
    pipes: list = field(default_factory=list)              # Pipe (horizontal)
    ellipsoids: list = field(default_factory=list)         # Ellipsoid (vessel, sphere)
    pedestals: list = field(default_factory=list)          # (name, xy, r, z_top): stands under the tanks
    saddles: list = field(default_factory=list)            # Box under the pipe
    markers: list = field(default_factory=list)            # ArucoMarker, one per surface
    wall_panels: list = field(default_factory=list)        # VerticalPanel (wavy heightfield on a wall)
    wall_planes: list = field(default_factory=list)        # WallPlane carrying a flat sticker

    # ------------------------------------------------------------------ surfaces (sim/markers.py interface)
    def surface(self, name):
        for s in self.cyl_surfaces + self.pipes + self.ellipsoids + self.wall_planes:
            if s.name == name:
                return s
        raise KeyError(name)

    def true_marker(self, mid):
        from markers import marker_frame
        mk = next(m for m in self.markers if m.id == mid)
        p, n, e_u, e_v = marker_frame(self, mk)
        return p, np.stack([e_u, e_v, n], 1)

    # ------------------------------------------------------------------ MuJoCo scene
    def add_to(self, spec: mujoco.MjSpec):
        super().add_to(spec)                    # terrain, walls, stairs, boxes, obstacle cylinders (tasks: none)
        wb = spec.worldbody

        def work(g):                            # touched by the force-control contact tip (bit 2), stiff steel
            g.group, g.contype, g.conaffinity, g.solref = 1, 3, 3, [0.005, 1.0]
        for c in self.cyl_surfaces:
            z0 = float(self.terrain.height(*c.xy)) - 0.02
            g = wb.add_geom()
            g.name, g.type, g.size = c.name, mujoco.mjtGeom.mjGEOM_CYLINDER, [c.radius, c.height / 2, 0]
            g.pos, g.rgba = [*c.xy, z0 + c.height / 2], c.rgba
            work(g)
        for p in self.pipes:
            g = wb.add_geom()
            g.name, g.type, g.size = p.name, mujoco.mjtGeom.mjGEOM_CYLINDER, [p.radius, 0, 0]
            g.fromto, g.rgba = [*p.end, *(p.end + p.length * p.axis)], p.rgba
            work(g)
            for k, end in enumerate((p.end, p.end + p.length * p.axis)):
                cap = wb.add_geom()
                cap.name, cap.type, cap.size = f"{p.name}_flange{k}", mujoco.mjtGeom.mjGEOM_CYLINDER, [p.radius + 0.02, 0, 0]
                cap.fromto = [*(end - 0.012 * p.axis), *(end + 0.012 * p.axis)]
                cap.rgba, cap.group = [0.85, 0.65, 0.1, 1], 1
        for e in self.ellipsoids:
            g = wb.add_geom()
            g.name, g.type, g.size, g.pos, g.rgba = e.name, mujoco.mjtGeom.mjGEOM_ELLIPSOID, list(e.radii), list(e.center), e.rgba
            work(g)
        for name, xy, r, z_top in self.pedestals:
            z0 = float(self.terrain.height(*xy)) - 0.02
            g = wb.add_geom()
            g.name, g.type, g.size = name, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, (z_top - z0) / 2, 0]
            g.pos, g.rgba, g.group = [*xy, (z_top + z0) / 2], [0.3, 0.3, 0.32, 1], 1
        for b in self.saddles:
            g = wb.add_geom()
            g.name, g.type, g.size, g.pos = b.name, mujoco.mjtGeom.mjGEOM_BOX, list(b.half), list(b.center)
            g.rgba, g.group = b.rgba, 1
        for pn in self.wall_panels:
            self._add_panel(spec, pn)
        from markers import add_markers
        add_markers(spec, self)

    # ------------------------------------------------------------------ map (what the robot knows)
    def footprints(self):
        return [fp for _, _, fp in self._map_entries()]

    def _map_entries(self):
        out = [(b.name, False, ("box", b.center[0], b.center[1], b.half[0], b.half[1], b.yaw)) for b in self.boxes]
        out += [(n, True, ("circle", xy[0], xy[1], r)) for n, xy, r, _, _ in self.cylinders]
        out += [(c.name, True, ("circle", c.xy[0], c.xy[1], c.radius)) for c in self.cyl_surfaces]
        for p in self.pipes:
            c = p.end + 0.5 * p.length * p.axis
            out.append((p.name, True, ("box", c[0], c[1], p.length / 2 + 0.02, p.radius + 0.02, p.yaw)))
        out += [(e.name, True, ("circle", e.center[0], e.center[1], max(e.radii[:2]))) for e in self.ellipsoids]
        for pn in self.wall_panels:              # the map knows a panel is mounted there (not its shape)
            n = np.asarray(pn.normal, float)
            c = np.asarray(pn.center, float) + (pn.standoff + pn.amp) / 2 * n
            out.append((pn.name, True, ("box", c[0], c[1], pn.hu + 0.03, (pn.standoff + pn.amp) / 2 + 0.01,
                                        float(np.arctan2(n[1], n[0])) + np.pi / 2)))
        return out

    def localiser_map(self):
        """(solid, exclude) footprints for sim/perception/map_localiser.py: solid = shaped the same at every height
        (crates, cabinet, barrels, column); exclude = only a height range of the footprint is occupied (pipe on saddles,
        tanks on pedestals, the panel on the wall) - depth points there are not matched."""
        solid = [("box", b.center[0], b.center[1], b.half[0], b.half[1], b.yaw) for b in self.boxes]
        solid += [("circle", xy[0], xy[1], r) for _, xy, r, _, _ in self.cylinders]
        solid += [("circle", c.xy[0], c.xy[1], c.radius) for c in self.cyl_surfaces]
        exclude = [fp for n, curved, fp in self._map_entries()
                   if n in {p.name for p in self.pipes} | {e.name for e in self.ellipsoids} | {p.name for p in self.wall_panels}]
        return solid, exclude

    def map_objects(self):
        """For sim/planning/explorer.py: dict(name, curved, footprint, kind) per map entry, plus the staircase."""
        st = self.stairs
        out = [dict(name=n, curved=c, footprint=fp, kind="obstacle") for n, c, fp in self._map_entries()]
        out.append(dict(name="stairs", curved=False, kind="stairs",
                        footprint=("box", (st.x_start + st.x_end) / 2, (st.y_lo + st.y_hi) / 2,
                                   (st.x_end - st.x_start) / 2, (st.y_hi - st.y_lo) / 2, 0.0)))
        return out


def curved_world(decimate=1, panel_seed=4):
    ter = Terrain(decimate)
    x0, x1, y0, y1 = -2.3, 22.3, -2.35, 2.35
    st = Stairs(x_start=8.6, y_lo=y0, y_hi=y1)
    g = lambda x, y: float(ter.height(x, y))
    boxes = [
        Box("crates_a", (1.4, -0.9, 0.35), (0.40, 0.45, 0.35)),
        Box("crates_b", (6.3, 1.2, 0.30), (0.35, 0.50, 0.30), yaw=0.25),
        Box("pallet", (0.4, 1.75, 0.08), (0.55, 0.45, 0.08), rgba=[0.55, 0.4, 0.22, 1]),
        Box("crates_c", (16.0, -1.0, 0.40), (0.45, 0.40, 0.40), yaw=-0.3),
        Box("cabinet", (20.6, -1.7, 0.55), (0.35, 0.30, 0.55), rgba=[0.2, 0.3, 0.55, 1]),
    ]
    cylinders = [("column", (12.8, 0.7), 0.22, 1.4, [0.5, 0.5, 0.52, 1]),        # plain obstacle (as in v8)
                 ("barrel_1", (7.4, -1.9), 0.20, 0.85, [0.15, 0.45, 0.25, 1]),
                 ("barrel_2", (16.4, 1.9), 0.20, 0.85, [0.6, 0.15, 0.12, 1])]
    vx, vy = 3.9, 1.75
    vessel = Ellipsoid("vessel", (vx, vy, g(vx, vy) + 0.62), (0.40, 0.40, 0.60), rgba=[0.30, 0.45, 0.55, 1])
    px, py = 4.4, -1.85
    pipe = Pipe("pipe", x0=px, y=py, z=g(px + 0.8, py) + 0.66, length=1.6, radius=0.16)
    saddles = [Box(f"saddle_{k}", (sx, py, (pipe.z - pipe.radius + 0.02) / 2), (0.06, 0.10, (pipe.z - pipe.radius + 0.02) / 2),
                   rgba=[0.85, 0.65, 0.1, 1]) for k, sx in enumerate((px + 0.2, px + 1.4))]
    tx, ty = 18.2, -1.55
    tank = Ellipsoid("sphere_tank", (tx, ty, g(tx, ty) + 0.66), (0.45, 0.45, 0.45), rgba=[0.75, 0.75, 0.72, 1])
    markers = [
        ArucoMarker(5, "vessel", MARKER_SIZE, azimuth_deg=-90.0, z=g(vx, vy - 0.4) + 0.40),
        ArucoMarker(23, "pipe", MARKER_SIZE, s=0.55, theta_deg=100.0),
        ArucoMarker(37, "sphere_tank", MARKER_SIZE, azimuth_deg=90.0, z=g(tx, ty + 0.4) + 0.40),
    ]
    # the wavy panel of v8 (marker 5 there): same place, same work order; its marker is a flat sticker on the wall
    wx = 14.4
    zm = g(wx, y0 + 0.7) + 0.40
    panel = VerticalPanel("panel_7", (wx, y0, zm + 0.30), (0, 1, 0), hv=0.14, seed=panel_seed)
    wall = WallPlane("wall_s_panel", (wx, y0, 0.0), (0.0, 1.0, 0.0))
    markers.append(ArucoMarker(7, wall.name, MARKER_SIZE, azimuth_deg=90.0, z=zm))
    return CurvedWorld(terrain=ter, bounds=(x0, x1, y0, y1), stairs=st, boxes=boxes, cylinders=cylinders, tasks=[],
                       cyl_surfaces=[], pipes=[pipe], ellipsoids=[vessel, tank],
                       pedestals=[("tank_pedestal", (tx, ty), 0.22, tank.center[2] - 0.40)], saddles=saddles,
                       markers=markers, wall_panels=[panel], wall_planes=[wall])
