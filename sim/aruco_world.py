"""ArUco work-order world on the user's terrain (assets/scene_terrain, from scene_terraine.blend).

    25 m x 5 m displaced floor (0-8 cm relief, heightfield) with walls around it
    a full-width staircase across the corridor (2 x 8 cm up, 1.2 m platform, 2 x 8 cm down)
    crates / a column / barrels as obstacles
    work sites, each tagged with an ArUco marker (DICT_4X4_50, 20 cm black square on a 25 cm white plate).
    Every work surface is VERTICAL (a flat wall, a vertical pipe, a wavy panel on a wall), so a forward-looking depth
    camera sees it face-on. The marker sits just below the work area, on (or bracketed to) the work surface.

A marker means only "work here": the work order says which rectangle of the surface to cover, in the marker frame
(x right, y up, z out of the plate): centre offset + size. It does NOT say what the surface is (wall, pipe, panel).

What the robot knows (the map, no SLAM): walls, staircase, obstacles - everything that builds the occupancy grid.
What it does NOT know: where the markers are, what the work surfaces look like, where exactly they are.
Markers are found by the head RGB camera (OpenCV ArUco + PnP), the surface by the head depth camera.
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

# work orders the robot carries: marker id -> area to cover (marker frame: x right, y up; centre offset + size [m])
# and the paint colour. Nothing about the surface's shape - the robot measures that.
WORK_ORDERS = {
    3: dict(area_offset=(0.0, 0.26), area_size=(0.75, 0.16), color=(0.85, 0.12, 0.10)),
    11: dict(area_offset=(0.0, 0.26), area_size=(0.14, 0.16), color=(0.10, 0.35, 0.90)),
    5: dict(area_offset=(0.0, 0.26), area_size=(0.40, 0.16), color=(0.10, 0.55, 0.20)),
    17: dict(area_offset=(0.0, 0.26), area_size=(0.14, 0.16), color=(0.95, 0.55, 0.05)),
    7: dict(area_offset=(0.0, 0.26), area_size=(0.40, 0.16), color=(0.55, 0.15, 0.75)),
}
# Fixed-site mode (no exploration yet): an operator's rough viewpoint per work order (map frame: x, y, facing yaw) and
# the visiting order. The robot still has to find and measure the marker and the surface itself from there.
SITE_HINTS = {
    3: (3.0, 1.0, np.pi / 2),
    11: (4.2, -0.85, 0.0),
    7: (9.8, 0.9, np.pi / 2),          # on the stair platform (the work band is too high to reach from the floor)
    5: (14.4, -1.0, -np.pi / 2),
    17: (17.8, 1.0, 0.0),
}
SITE_ORDER = [3, 11, 7, 5, 17]
MARKER_Z = 0.40             # marker centre above the local ground: work band 0.58-0.74 m (the Z1's reach band with the
                            # tool along the normal of a vertical surface, from a Go2 standing on the same ground)


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
    mount: str = "wall"      # "wall": sticker on the surface | "bracket": plate on a short bracket off a pipe

    def frame(self):
        """Marker frame (columns x right, y up, z out of the plate), as OpenCV's ArUco pose convention."""
        z = np.asarray(self.normal, float); z /= np.linalg.norm(z)
        y = np.array([0, 0, 1.0])
        x = np.cross(y, z)
        return np.stack([x, y, z], 1)


@dataclass
class VerticalPanel:
    """A random wavy panel (no closed-form shape: Gaussian bumps + a wave, sim/world.py FreeformPanel) standing
    vertically on a wall: centre on the wall face, outward normal n (horizontal), 2 hu wide, 2 hv tall."""
    name: str
    center: tuple
    normal: tuple
    hu: float = 0.25
    hv: float = 0.15
    amp: float = 0.03
    standoff: float = 0.05   # mean surface distance in front of the wall face
    seed: int = 0
    res: float = 0.004
    rgba: list = field(default_factory=lambda: [0.55, 0.57, 0.6, 1])

    def frame(self):
        z = np.asarray(self.normal, float); z /= np.linalg.norm(z)
        y = np.array([0, 0, 1.0])
        return np.stack([np.cross(y, z), y, z], 1)


@dataclass
class Task:                  # ground truth (scene + scoring only)
    marker: Marker
    pipe: VerticalPipe = None
    panel: VerticalPanel = None


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
            if task.panel is not None:
                self._add_panel(spec, task.panel)
            self._add_marker(spec, task.marker)

    def _add_panel(self, spec, pn: VerticalPanel):
        from world import FreeformPanel
        shape = FreeformPanel(pn.name, 0.0, 0.0, pn.hu, pn.hv, 0.0, amp=pn.amp, seed=pn.seed)
        ncol, nrow = int(round(2 * pn.hu / pn.res)) + 1, int(round(2 * pn.hv / pn.res)) + 1
        U, V = np.meshgrid(np.linspace(-pn.hu, pn.hu, ncol), np.linspace(-pn.hv, pn.hv, nrow))
        Hh = shape.height(U, V) + pn.standoff
        lo, hi = Hh.min(), Hh.max()
        hf = spec.add_hfield()
        hf.name, hf.nrow, hf.ncol = pn.name, nrow, ncol
        hf.size = [pn.hu, pn.hv, hi - lo, 0.01]
        hf.userdata = ((Hh - lo) / (hi - lo)).ravel().tolist()
        R = pn.frame()                          # hfield local x -> along the wall, y -> up, z -> out of the wall
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
        g = spec.worldbody.add_geom()
        g.name, g.type, g.hfieldname = pn.name, mujoco.mjtGeom.mjGEOM_HFIELD, pn.name
        g.pos, g.quat, g.rgba, g.group = list(np.asarray(pn.center) + lo * R[:, 2]), q, pn.rgba, 1
        g.contype = g.conaffinity = 3           # touched by the contact tip
        g.solref = [0.005, 1.0]
        fr = spec.worldbody.add_geom()          # frame around the panel, flush with the wall
        fr.name, fr.type = pn.name + "_frame", mujoco.mjtGeom.mjGEOM_BOX
        fr.size, fr.pos, fr.quat = [pn.hu + 0.03, pn.hv + 0.03, 0.005], list(np.asarray(pn.center) + 0.005 * R[:, 2]), q
        fr.rgba, fr.group = [0.3, 0.3, 0.32, 1], 1

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
        if mk.mount == "bracket":                # backing board + a short arm back to the pipe behind it
            p = np.asarray(mk.pos) - 0.008 * n
            b = wb.add_geom()
            b.name, b.type, b.size = f"tag_{mk.id}", mujoco.mjtGeom.mjGEOM_BOX, [0.005, PLATE_SIZE / 2 + 0.01, PLATE_SIZE / 2 + 0.01]
            b.pos, b.quat, b.rgba, b.group = list(p), q, PAINT_YELLOW, 1
            b.contype = b.conaffinity = 0
            arm = wb.add_geom()
            arm.name, arm.type, arm.size = f"bracket_{mk.id}", mujoco.mjtGeom.mjGEOM_BOX, [0.03, 0.015, 0.015]
            arm.pos, arm.quat, arm.rgba, arm.group = list(p - 0.03 * n), q, [0.25, 0.25, 0.27, 1], 1
            arm.contype = arm.conaffinity = 0

    def true_marker(self, mid):
        """Ground truth (scene drawing + scoring only): marker centre and frame (columns x right, y up, z out)."""
        mk = next(t.marker for t in self.tasks if t.marker.id == mid)
        return np.asarray(mk.pos, float), mk.frame()

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


def aruco_world(decimate=1, pipe_radius=(0.11, 0.13), panel_seed=4):
    """The default work-order world. pipe_radius / panel_seed: the real work surfaces (unknown to the robot)."""
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
    tasks = []
    # flat wall band on the north wall
    tasks.append(Task(Marker(3, (3.0, y1, hz(3.0, y1 - 0.7, MARKER_Z)), (0, -1, 0))))
    # vertical pipe, tagged by a plate bracketed to its front (facing back down the corridor)
    for mid, xy, r in ((11, (5.55, -0.85), pipe_radius[0]), (17, (19.1, 1.0), pipe_radius[1])):
        n = np.array([-1.0, 0, 0])
        pos = np.array([xy[0], xy[1], 0.0]) + (r + 0.045) * n
        tasks.append(Task(Marker(mid, (pos[0], pos[1], hz(pos[0], pos[1], MARKER_Z)), tuple(n), mount="bracket"),
                          pipe=VerticalPipe(f"pipe_{mid}", xy, r, 0.95)))
    # wavy panel on the south wall, marker on the wall just below it
    zm = hz(14.4, y0 + 0.7, MARKER_Z)
    tasks.append(Task(Marker(5, (14.4, y0, zm), (0, 1, 0)),
                      panel=VerticalPanel("panel_5", (14.4, y0, zm + 0.30), (0, 1, 0), hv=0.14, seed=panel_seed)))
    # a wavy panel high on the north wall beside the stair platform: its work band is 0.76-0.92 m above the floor,
    # above what the arm reaches from the floor - the robot has to stand on the platform (0.20 m up) to work on it
    zp = st.base_z + st.n * st.rise
    xm = (st.x_start + st.x_end) / 2
    tasks.append(Task(Marker(7, (xm, y1, zp + MARKER_Z), (0, -1, 0)),
                      panel=VerticalPanel("panel_7", (xm, y1, zp + MARKER_Z + 0.30), (0, -1, 0), hv=0.14, seed=2)))
    return ArucoWorld(terrain=ter, bounds=(x0, x1, y0, y1), stairs=st, boxes=boxes, cylinders=cylinders, tasks=tasks)
