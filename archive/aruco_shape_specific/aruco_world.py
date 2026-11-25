"""ArUco work-order world on the user's terrain (assets/scene_terrain, from scene_terraine.blend).

    25 m x 5 m displaced floor (0-8 cm relief, heightfield) with walls around it
    a full-width staircase across the corridor (2 x 8 cm up, 1.2 m platform, 2 x 8 cm down)
    crates / a column / barrels as obstacles
    work sites, each tagged with an ArUco marker (DICT_4X4_50, 20 cm black square on a 25 cm white plate):
        "paint" markers on a wall: spray-paint the rectangle next to the marker (marker frame: area offset/size)
        "pipe"  markers on a sign post: a vertical pipe stands next to the sign; trace its surface under contact force

What the robot knows (the map, no SLAM): walls, staircase, obstacles - everything that builds the occupancy grid.
What it does NOT know: where the markers are, where the pipes are, their radius, where the wall face is exactly.
Markers are found by the head RGB camera (OpenCV ArUco + PnP), the pipe / wall geometry by the head depth camera.
The `tasks` list here is only the ground truth used to build the scene and to score the result.

Geom groups (as in sim/world.py): terrain + stairs = group 0 (the locomotion height scan sees them),
walls / obstacles / workpieces / markers = group 1 (depth camera sees them), robot = groups 2/3.
Pipes also have collision bit 2 (touched by the force-control contact tip).
"""
import json
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERRAIN_DIR = os.path.join(ROOT, "assets", "scene_terrain")
MARKER_DIR = os.path.join(ROOT, "assets", "aruco")
ARUCO_DICT = "DICT_4X4_50"
MARKER_SIZE = 0.20      # black square edge [m] (what PnP needs)
PLATE_SIZE = 0.25       # with the white quiet zone
STEEL, CRATE, PAINT_YELLOW = [0.45, 0.47, 0.5, 1], [0.62, 0.45, 0.25, 1], [0.85, 0.65, 0.1, 1]
WALL_RGBA = [0.78, 0.78, 0.76, 1]

# work orders the robot carries: marker id -> what to do there (the robot does not know where the markers are)
#   paint: trace a band of the wall in contact (area in the marker frame: x right, y up, origin = marker centre:
#          centre offset + size [m]). Wider than one arm reach -> split into patches, the robot sidesteps.
#   pipe : a vertical pipe stands somewhere within `search_radius` of the sign; trace +-arc_deg x a height band
#          (above the local ground) in contact
WORK_ORDERS = {
    3: dict(kind="paint", area_offset=(0.65, -0.28), area_size=(0.75, 0.16), color=(0.85, 0.12, 0.10)),
    11: dict(kind="pipe", search_radius=1.3, arc_deg=35.0, band=(0.40, 0.56), color=(0.10, 0.35, 0.90)),
    5: dict(kind="paint", area_offset=(0.65, -0.28), area_size=(0.75, 0.16), color=(0.10, 0.55, 0.20)),
    17: dict(kind="pipe", search_radius=1.3, arc_deg=35.0, band=(0.40, 0.56), color=(0.95, 0.55, 0.05)),
}
PATCH_MAX_WIDTH = 0.30      # widest patch the Z1 traces from one stance (0.30 x 0.16 m, joint margin >= 20 %)


# ----------------------------------------------------------------------------------------------- terrain
class Terrain:
    """The user's displaced floor (heightfield, 2 cm cells) + height queries. `decimate` > 1 averages k x k cells
    (fewer render triangles; collision and the policy's height scan use the same, coarser surface)."""

    def __init__(self, decimate=1):
        self.info = json.load(open(os.path.join(TERRAIN_DIR, "terrain.json")))
        H = np.load(os.path.join(TERRAIN_DIR, "heights.npy")).astype(float)
        if decimate > 1:
            k = decimate
            ny, nx = (H.shape[0] - 1) // k * k + 1, (H.shape[1] - 1) // k * k + 1
            H = H[:ny:k, :nx:k]
        self.H = H
        self.x0, self.y0 = self.info["x0"], self.info["y0"]
        self.x1 = self.x0 + (H.shape[1] - 1) * self.info["cell"] * max(decimate, 1)
        self.y1 = self.y0 + (H.shape[0] - 1) * self.info["cell"] * max(decimate, 1)
        self.cell = self.info["cell"] * max(decimate, 1)

    def height(self, x, y):
        """Bilinear terrain height (world z) at (x, y); 0 outside."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        u = np.clip((x - self.x0) / self.cell, 0, self.H.shape[1] - 1.001)
        v = np.clip((y - self.y0) / self.cell, 0, self.H.shape[0] - 1.001)
        i, j = np.floor(v).astype(int), np.floor(u).astype(int)
        fu, fv = u - j, v - i
        H = self.H
        h = (H[i, j] * (1 - fu) * (1 - fv) + H[i, j + 1] * fu * (1 - fv) + H[i + 1, j] * (1 - fu) * fv
             + H[i + 1, j + 1] * fu * fv)
        inside = (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)
        return np.where(inside, h, 0.0)

    def add_to(self, spec, albedo):
        wb = spec.worldbody
        zmin, zmax = float(self.H.min()), float(self.H.max())
        ny, nx = self.H.shape
        hf = spec.add_hfield()
        hf.name, hf.nrow, hf.ncol = "terrain_hf", ny, nx
        rx, ry = (self.x1 - self.x0) / 2, (self.y1 - self.y0) / 2
        hf.size = [rx, ry, zmax - zmin, 0.05]
        hf.userdata = ((self.H - zmin) / (zmax - zmin)).ravel().tolist()
        g = wb.add_geom()
        g.name, g.type, g.hfieldname = "terrain", mujoco.mjtGeom.mjGEOM_HFIELD, "terrain_hf"
        g.pos, g.group = [(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, zmin], 0
        g.friction = [1.0, 0.02, 0.01]
        if albedo is not None:
            t = spec.add_texture()
            t.name, t.type, t.file = "terrain_tex", mujoco.mjtTexture.mjTEXTURE_2D, albedo
            m = spec.add_material()
            m.name = "terrain_mat"
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
            m.texrepeat, m.texuniform = [round(2 * rx / 5.0), round(2 * ry / 5.0)], False
            g.material = "terrain_mat"
        else:
            g.rgba = [0.5, 0.48, 0.44, 1]


# ----------------------------------------------------------------------------------------------- objects
@dataclass
class Box:
    name: str
    center: tuple
    half: tuple
    yaw: float = 0.0
    rgba: list = field(default_factory=lambda: CRATE)
    group: int = 1


@dataclass
class Stairs:
    """Full-width staircase along +x: n steps up, a platform, n steps down. The planner treats it as a portal
    (crossed straight, perpendicular to the steps) - see sim/planning/task_planner.py."""
    x_start: float           # foot of the first riser
    y_lo: float
    y_hi: float
    n: int = 2
    rise: float = 0.08
    run: float = 0.30
    platform: float = 1.2
    base_z: float = 0.04     # mean terrain height under the stairs

    @property
    def x_end(self):
        return self.x_start + 2 * self.n * self.run + self.platform

    def blocks(self):
        out = []
        for k in range(self.n):                  # up: step k top at base + (k + 1) rise, extends to the platform
            x_a = self.x_start + k * self.run
            x_b = self.x_end - k * self.run
            top = self.base_z + (k + 1) * self.rise
            out.append(((x_a + x_b) / 2, (x_b - x_a) / 2, top))
        return out

    def height(self, x):
        """Walking-surface height above the stairs (None outside)."""
        if not (self.x_start <= x <= self.x_end):
            return None
        k = min(self.n, int((x - self.x_start) / self.run) + 1, int((self.x_end - x) / self.run) + 1)
        return self.base_z + k * self.rise


@dataclass
class VerticalPipe:
    name: str
    xy: tuple
    radius: float
    height: float
    rgba: list = field(default_factory=lambda: STEEL)


@dataclass
class Marker:
    id: int
    pos: tuple               # plate centre (on the surface it is stuck to)
    normal: tuple            # outward (horizontal)
    post: bool = False       # on its own sign post (else stuck to a wall)

    def frame(self):
        """Marker frame (columns x right, y up, z out of the plate), as OpenCV's ArUco pose convention."""
        z = np.asarray(self.normal, float); z /= np.linalg.norm(z)
        y = np.array([0, 0, 1.0])
        x = np.cross(y, z)
        return np.stack([x, y, z], 1)


@dataclass
class Task:                  # ground truth (scene + scoring only)
    marker: Marker
    pipe: VerticalPipe = None


@dataclass
class ArucoWorld:
    terrain: Terrain
    bounds: tuple                              # inner wall faces (x0, x1, y0, y1)
    stairs: Stairs
    boxes: list
    cylinders: list                            # round obstacles (map-known): (name, xy, r, h, rgba)
    tasks: list
    start: tuple = (-1.6, 0.0, 0.0)
    wall_h: float = 1.4

    # ------------------------------------------------------------------ MuJoCo scene
    def add_to(self, spec: mujoco.MjSpec):
        wb = spec.worldbody
        for g in list(spec.geoms):
            if g.name == "floor":
                spec.delete(g)
        spec.visual.headlight.ambient = [0.22, 0.22, 0.24]
        spec.visual.headlight.diffuse = [0.32, 0.32, 0.32]
        spec.visual.quality.shadowsize = 4096
        sun = wb.add_light()
        sun.name, sun.pos, sun.dir = "sun", [10, -10, 20], [0.15, 0.4, -0.9]
        sun.type, sun.diffuse, sun.castshadow = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, [0.5, 0.48, 0.45], True

        apron = wb.add_geom()   # flat surround below the terrain edge
        apron.name, apron.type, apron.size = "apron", mujoco.mjtGeom.mjGEOM_PLANE, [40, 40, 0.1]
        apron.pos, apron.rgba, apron.group = [10, 0, -0.002], [0.42, 0.42, 0.42, 1], 0
        albedo = os.path.join(TERRAIN_DIR, "albedo_2k.png")
        self.terrain.add_to(spec, albedo if os.path.exists(albedo) else None)

        def box(b: Box, z_lo=None):
            g = wb.add_geom()
            g.name, g.type = b.name, mujoco.mjtGeom.mjGEOM_BOX
            g.size, g.pos = list(b.half), list(b.center)
            g.quat = [np.cos(b.yaw / 2), 0, 0, np.sin(b.yaw / 2)]
            g.rgba, g.group = b.rgba, b.group
            return g

        x0, x1, y0, y1 = self.bounds
        t, h = 0.1, self.wall_h
        for b in [Box("wall_n", ((x0 + x1) / 2, y1 + t, h / 2 - 0.05), ((x1 - x0) / 2 + 2 * t, t, h / 2 + 0.05)),
                  Box("wall_s", ((x0 + x1) / 2, y0 - t, h / 2 - 0.05), ((x1 - x0) / 2 + 2 * t, t, h / 2 + 0.05)),
                  Box("wall_w", (x0 - t, (y0 + y1) / 2, h / 2 - 0.05), (t, (y1 - y0) / 2, h / 2 + 0.05)),
                  Box("wall_e", (x1 + t, (y0 + y1) / 2, h / 2 - 0.05), (t, (y1 - y0) / 2, h / 2 + 0.05))]:
            b.rgba = WALL_RGBA
            g = box(b)
            g.contype = g.conaffinity = 3          # walls are work surfaces too: touched by the contact tip
            g.solref = [0.005, 1.0]                # concrete: stiff contact (as the steel pipe)
        st = self.stairs
        for k, (cx, hx, top) in enumerate(st.blocks()):
            b = Box(f"stair_{k}", (cx, (st.y_lo + st.y_hi) / 2, (top - 0.03) / 2), (hx, (st.y_hi - st.y_lo) / 2, (top + 0.03) / 2),
                    rgba=[0.55, 0.55, 0.53, 1], group=0)
            box(b).friction = [1.0, 0.02, 0.01]
            nose = Box(f"stair_nose_{k}", (cx - hx + 0.015, (st.y_lo + st.y_hi) / 2, top - 0.004), (0.015, (st.y_hi - st.y_lo) / 2, 0.0045),
                       rgba=PAINT_YELLOW, group=0)
            g = box(nose); g.contype = g.conaffinity = 0
            nose.center = (cx + hx - 0.015, nose.center[1], nose.center[2]); nose.name = f"stair_nose_b{k}"
            g = box(nose); g.contype = g.conaffinity = 0
        for b in self.boxes:
            box(b)
        for name, xy, r, hh, rgba in self.cylinders:
            g = wb.add_geom()
            g.name, g.type, g.size = name, mujoco.mjtGeom.mjGEOM_CYLINDER, [r, hh / 2, 0]
            g.pos, g.rgba, g.group = [*xy, hh / 2 - 0.02], rgba, 1
        for task in self.tasks:
            if task.pipe is not None:
                p = task.pipe
                z0 = float(self.terrain.height(*p.xy)) - 0.02
                g = wb.add_geom()
                g.name, g.type, g.size = p.name, mujoco.mjtGeom.mjGEOM_CYLINDER, [p.radius, p.height / 2, 0]
                g.pos, g.rgba, g.group = [*p.xy, z0 + p.height / 2], p.rgba, 1
                g.contype = g.conaffinity = 3          # also touched by the contact tip
                g.solref = [0.005, 1.0]
                fl = wb.add_geom()                      # base flange
                fl.name, fl.type, fl.size = p.name + "_flange", mujoco.mjtGeom.mjGEOM_CYLINDER, [p.radius + 0.05, 0.03, 0]
                fl.pos, fl.rgba, fl.group = [*p.xy, z0 + 0.03], PAINT_YELLOW, 1
                cap = wb.add_geom()
                cap.name, cap.type, cap.size = p.name + "_cap", mujoco.mjtGeom.mjGEOM_CYLINDER, [p.radius + 0.01, 0.015, 0]
                cap.pos, cap.rgba, cap.group = [*p.xy, z0 + p.height], [0.3, 0.3, 0.32, 1], 1
            self._add_marker(spec, task.marker)

    def _add_marker(self, spec, mk: Marker):
        wb = spec.worldbody
        n = np.asarray(mk.normal, float); n /= np.linalg.norm(n)
        yaw = np.arctan2(-n[1], -n[0])          # plate local -x face looks along n (texture pre-flipped for it)
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        tex = spec.add_texture()
        tex.name, tex.type, tex.file = f"aruco_{mk.id}", mujoco.mjtTexture.mjTEXTURE_2D, marker_png(mk.id)
        mat = spec.add_material()
        mat.name = f"aruco_{mk.id}"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.specular, mat.shininess = 0.0, 0.0
        g = wb.add_geom()
        g.name, g.type, g.size = f"marker_{mk.id}", mujoco.mjtGeom.mjGEOM_BOX, [0.003, PLATE_SIZE / 2, PLATE_SIZE / 2]
        g.pos, g.quat, g.material, g.group = list(np.asarray(mk.pos) + 0.003 * n), q, mat.name, 1
        g.contype = g.conaffinity = 0
        if mk.post:                              # sign post + backing board behind the plate
            p = np.asarray(mk.pos) - 0.02 * n
            b = wb.add_geom()
            b.name, b.type, b.size = f"sign_{mk.id}", mujoco.mjtGeom.mjGEOM_BOX, [0.015, PLATE_SIZE / 2 + 0.03, PLATE_SIZE / 2 + 0.03]
            b.pos, b.quat, b.rgba, b.group = list(p), q, PAINT_YELLOW, 1
            pz = mk.pos[2] - PLATE_SIZE / 2 - 0.03
            s = wb.add_geom()
            s.name, s.type, s.size = f"post_{mk.id}", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.025, pz / 2 + 0.02, 0]
            s.pos, s.rgba, s.group = [p[0] - 0.03 * n[0], p[1] - 0.03 * n[1], pz / 2 - 0.02], [0.25, 0.25, 0.27, 1], 1

    # ------------------------------------------------------------------ map (what the robot knows)
    def footprints(self):
        """Obstacle footprints for the occupancy grid. Stairs are NOT here: they are a portal (task_planner)."""
        out = [("box", b.center[0], b.center[1], b.half[0], b.half[1], b.yaw) for b in self.boxes]
        out += [("circle", xy[0], xy[1], r) for _, xy, r, _, _ in self.cylinders]
        return out

    def ground_z(self, x, y):
        h = self.stairs.height(x) if self.stairs.y_lo <= y <= self.stairs.y_hi else None
        return float(self.terrain.height(x, y)) if h is None else h


def marker_png(mid, px=200):
    """ArUco marker image with a white quiet zone, horizontally pre-flipped (MuJoCo maps 2-D textures mirrored on the
    box face used for the plate). Cached in assets/aruco/."""
    import cv2
    os.makedirs(MARKER_DIR, exist_ok=True)
    path = os.path.join(MARKER_DIR, f"aruco_{ARUCO_DICT}_{mid}.png")
    if not os.path.exists(path):
        D = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
        img = cv2.aruco.generateImageMarker(D, mid, px)
        b = int(round(px * (PLATE_SIZE - MARKER_SIZE) / (2 * MARKER_SIZE)))
        img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=255)[:, ::-1]
        cv2.imwrite(path, np.stack([img] * 3, -1))
    return path


def aruco_world(decimate=1, pipe_radius=(0.11, 0.13)):
    """The default work-order world. pipe_radius: real radii of the two pipes (unknown to the robot)."""
    ter = Terrain(decimate)
    x0, x1, y0, y1 = -2.3, 22.3, -2.35, 2.35
    st = Stairs(x_start=8.6, y_lo=y0, y_hi=y1)
    hz = lambda x, y, h: float(ter.height(x, y)) + h
    boxes = [
        Box("crates_a", (1.4, -0.9, 0.35), (0.40, 0.45, 0.35)),
        Box("crates_b", (6.3, 1.2, 0.30), (0.35, 0.50, 0.30), yaw=0.25),
        Box("pallet", (0.4, 1.75, 0.08), (0.55, 0.45, 0.08), rgba=[0.55, 0.4, 0.22, 1]),
        Box("crates_c", (16.0, -1.0, 0.40), (0.45, 0.40, 0.40), yaw=-0.3),
        Box("cabinet", (20.6, -1.7, 0.55), (0.35, 0.30, 0.55), rgba=[0.2, 0.3, 0.55, 1]),
    ]
    cylinders = [("column", (12.8, 0.7), 0.22, 1.4, [0.5, 0.5, 0.52, 1]),
                 ("barrel_1", (7.4, -1.9), 0.20, 0.85, [0.15, 0.45, 0.25, 1]),
                 ("barrel_2", (16.4, 1.9), 0.20, 0.85, [0.6, 0.15, 0.12, 1])]
    tasks = [
        # wall paint on the north wall, marker 0.75 m up
        Task(Marker(3, (2.6, y1, 0.78), (0, -1, 0))),
        # vertical pipe beside a sign that faces back down the corridor
        Task(Marker(11, (5.3, -1.85, 0.62), (-1, 0, 0), post=True), VerticalPipe("pipe_a", (5.55, -0.85), pipe_radius[0], 0.95)),
        # beyond the stairs: wall paint on the south wall, then a second pipe
        Task(Marker(5, (14.4, y0, 0.78), (0, 1, 0))),
        Task(Marker(17, (18.8, 1.95, 0.62), (-1, 0, 0), post=True), VerticalPipe("pipe_b", (19.1, 1.0), pipe_radius[1], 0.95)),
    ]
    for t in tasks:  # sign posts stand on the terrain: lift plate heights by the local ground
        if t.marker.post:
            t.marker.pos = (t.marker.pos[0], t.marker.pos[1], hz(t.marker.pos[0], t.marker.pos[1], t.marker.pos[2]))
    return ArucoWorld(terrain=ter, bounds=(x0, x1, y0, y1), stairs=st, boxes=boxes, cylinders=cylinders, tasks=tasks)
