"""MuJoCo version of assets/myscene.blend: LCU-1600-class ship ('body', 132 m), two scanned rock ground patches and a
dock. Converted by scripts/convert_myscene.py -> assets/myscene_export/.

All scene meshes are visual only. The robot walks on an invisible flat collision floor at the rock patches' median
surface height (z = 0.10 m), i.e. no heightfield: feet dip a few cm into the higher rock bumps.

World (from the .blend, metres): ship x -15.6..14.9, y -69..63 (hull down to z -5.4);
rocks x 17.5..37.8, y -4..6, z -0.17..0.53; dock x 7..26, y 20..70.
"""
import json
import os

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT = os.path.join(ROOT, "assets", "myscene_export")
# Rock scan patches scaled to ROCK_SCALE so the stones are Go2-sized pebbles (6-15 cm) instead of 20-50 cm boulders;
# the texture follows the mesh via its UVs. Decimated LOD meshes (scripts/decimate_myscene_rocks.py) are tiled over the
# original rock area. Collision floor at the scaled patches' median surface height.
ROCK_SCALE = 0.3
FLOOR_Z = 0.10 * ROCK_SCALE
ROCKS_X, ROCKS_Y = (17.5, 37.8), (-4.0, 6.0)

# Ship (LCU, 132 m) scaled 0.75x about its rock-side edge (x = 14.9), so it stays alongside the rocks.
SHIP_SCALE, SHIP_ANCHOR = 0.75, (14.9, 0.0, 0.0)
# Dock scaled so its deck is 0.8 m high (4.9 m in the .blend), placed against the ship side between ship and rocks.
DOCK_ORIG_LO, DOCK_ORIG_HI = (7.175, 20.397, 0.058), (26.377, 69.858, 4.897)
DOCK_TOP = 0.8
DOCK_SCALE = DOCK_TOP / DOCK_ORIG_HI[2]
DOCK_LO = (15.3, -3.0, 0.0)
DOCK_HI = tuple(DOCK_LO[i] + DOCK_SCALE * (DOCK_ORIG_HI[i] - DOCK_ORIG_LO[i]) for i in range(3))
# 8 steps (10 cm rise, 30 cm run) from the rocks up to the dock deck, ascending towards -x, lane centred at y = 1
STAIR_N, STAIR_RUN, STAIR_Y, STAIR_W = 8, 0.30, 1.0, 1.2
STAIR_RISE = DOCK_TOP / STAIR_N
DOCKYARD_TEX = os.path.join(ROOT, "assets", "dockyard", "textures")


def add_myscene(spec):
    meta = json.load(open(os.path.join(EXPORT, "meta.json")))
    wb = spec.worldbody
    for g in list(spec.geoms):  # drop the robot scene's checker floor
        if g.name == "floor":
            spec.delete(g)
    spec.visual.headlight.ambient = [0.4, 0.4, 0.42]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.map.zfar = 400
    sun = wb.add_light()
    sun.name, sun.pos, sun.dir, sun.type = "sun", [30, -30, 60], [-0.4, 0.4, -0.8], mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse, sun.castshadow = [0.7, 0.68, 0.64], True

    floor = wb.add_geom()  # invisible collision floor (group 3 is hidden in the default renderer)
    floor.name, floor.type, floor.size, floor.pos = "walk_floor", mujoco.mjtGeom.mjGEOM_PLANE, [60, 60, 0.1], [0, 0, FLOOR_Z]
    floor.group, floor.rgba, floor.friction = 3, [0.5, 0.5, 0.5, 0], [1.0, 0.02, 0.01]

    mats = set()
    for mm in meta["meshes"]:
        if (mm["object"] or "").startswith("ground_rocks"):
            continue  # replaced by the scaled, tiled LOD patches below
        name = os.path.splitext(mm["obj"])[0]
        obj = mm["object"] or ""
        me = spec.add_mesh(); me.name, me.file = name, os.path.join(EXPORT, mm["obj"])
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = wb.add_geom()
        g.name, g.type, g.meshname = name, mujoco.mjtGeom.mjGEOM_MESH, name
        g.contype = g.conaffinity = 0
        if obj == "body":       # ship: v' = s v + (1 - s) anchor
            me.scale = [SHIP_SCALE] * 3
            g.pos = [(1 - SHIP_SCALE) * a for a in SHIP_ANCHOR]
        elif obj == "Dock_B001":  # dock: map its original lower corner to DOCK_LO
            me.scale = [DOCK_SCALE] * 3
            g.pos = [DOCK_LO[i] - DOCK_SCALE * DOCK_ORIG_LO[i] for i in range(3)]
        if mm["texture"]:
            mat = "mat_" + os.path.splitext(mm["texture"])[0]
            if mat not in mats:
                t = spec.add_texture()
                t.name, t.type, t.file = mat + "_tex", mujoco.mjtTexture.mjTEXTURE_2D, os.path.join(EXPORT, mm["texture"])
                m = spec.add_material(); m.name = mat
                m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
                mats.add(mat)
            g.material = mat
        else:
            g.rgba = mm["rgba"]
    _add_rock_tiles(spec, wb)
    stair_x0 = _add_dock_stairs(spec, wb)
    return dict(floor_z=FLOOR_Z, rocks_x=ROCKS_X, rocks_y=ROCKS_Y, rock_scale=ROCK_SCALE, dock_lo=DOCK_LO,
                dock_hi=DOCK_HI, stair_x0=stair_x0, stair_y=STAIR_Y)


def _tex_material(spec, name, file, repeat=(1, 1)):
    t = spec.add_texture()
    t.name, t.type, t.file = name + "_tex", mujoco.mjtTexture.mjTEXTURE_2D, file
    m = spec.add_material(); m.name = name
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
    m.texuniform, m.texrepeat = True, list(repeat)
    return name


def _box(wb, pos, half, material=None, rgba=None, contact=True, group=0):
    g = wb.add_geom()
    g.type, g.pos, g.size, g.group = mujoco.mjtGeom.mjGEOM_BOX, list(pos), list(half), group
    if material:
        g.material = material
    if rgba is not None:
        g.rgba = list(rgba)
    if not contact:
        g.contype = g.conaffinity = 0
    else:
        g.friction = [1.0, 0.02, 0.01]
    return g


def _add_dock_stairs(spec, wb):
    """Collision for the scaled dock deck + an 8-step textured stair from the rocks up to it (ascending towards -x)."""
    grate = _tex_material(spec, "stair_grate", os.path.join(DOCKYARD_TEX, "metal_grate_rusty.png"), (2, 2))
    conc = _tex_material(spec, "stair_concrete", os.path.join(DOCKYARD_TEX, "painted_concrete.png"), (1, 1))
    # dock deck: invisible collision slab under the (visual) scaled dock mesh
    _box(wb, [(DOCK_LO[0] + DOCK_HI[0]) / 2, (DOCK_LO[1] + DOCK_HI[1]) / 2, DOCK_TOP - 0.05],
         [(DOCK_HI[0] - DOCK_LO[0]) / 2, (DOCK_HI[1] - DOCK_LO[1]) / 2, 0.05], rgba=[0, 0, 0, 0], group=3)
    x_top = DOCK_HI[0]                       # stair meets the dock edge (rock side)
    x0 = x_top + STAIR_N * STAIR_RUN         # first step front edge, on the rocks
    hw = STAIR_W / 2
    for i in range(STAIR_N):
        zt = STAIR_RISE * (i + 1)
        xc = x0 - STAIR_RUN * i - STAIR_RUN / 2
        _box(wb, [xc, STAIR_Y, zt - 0.02], [STAIR_RUN / 2, hw, 0.02], grate)                        # tread
        _box(wb, [x0 - STAIR_RUN * i - 0.01, STAIR_Y, zt / 2], [0.01, hw, zt / 2], conc)            # riser
        for ys in (STAIR_Y - hw - 0.03, STAIR_Y + hw + 0.03):                                       # side walls
            _box(wb, [xc, ys, zt / 2], [STAIR_RUN / 2, 0.03, zt / 2], conc)
    for ys in (STAIR_Y - hw - 0.03, STAIR_Y + hw + 0.03):  # handrails (visual)
        r = wb.add_geom()
        r.type, r.size, r.fromto = mujoco.mjtGeom.mjGEOM_CAPSULE, [0.025, 0, 0], [x0, ys, 0.9, x_top, ys, DOCK_TOP + 0.9]
        r.rgba, r.contype, r.conaffinity = [0.9, 0.75, 0.1, 1], 0, 0
    return x0


def _add_rock_tiles(spec, wb):
    """Tile the two decimated rock patches (12.3 x 9.9 m originals -> 3.7 x 3.0 m at 0.3x) over the original rock area,
    alternating patch and yaw (0 / 180 deg) so the repetition is less visible."""
    tex = {"a": "packed_albedo.png", "b": "packed_albedo_001.png"}
    for k, f in tex.items():
        me = spec.add_mesh()
        me.name, me.file = f"rocks_{k}", os.path.join(EXPORT, f"ms_rocks_{k}_lod.obj")
        me.scale, me.inertia = [ROCK_SCALE] * 3, mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        t = spec.add_texture()
        t.name, t.type, t.file = f"rocks_{k}_tex", mujoco.mjtTexture.mjTEXTURE_2D, os.path.join(EXPORT, f)
        m = spec.add_material(); m.name = f"rocks_{k}_mat"
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
    sx, sy = 12.28 * ROCK_SCALE * 0.8, 9.94 * ROCK_SCALE * 0.8     # 20 % overlap: the scan edges are ragged
    nx = int((ROCKS_X[1] - ROCKS_X[0]) / sx) + 1
    ny = int((ROCKS_Y[1] - ROCKS_Y[0]) / sy) + 1
    for i in range(nx):
        for j in range(ny):
            k = "a" if (i + j) % 2 == 0 else "b"
            g = wb.add_geom()
            g.name, g.type, g.meshname, g.material = f"rocks_{i}_{j}", mujoco.mjtGeom.mjGEOM_MESH, f"rocks_{k}", f"rocks_{k}_mat"
            g.pos = [ROCKS_X[0] + sx / 2 + i * sx, ROCKS_Y[0] + sy / 2 + j * sy, 0.0]
            g.quat = [1, 0, 0, 0] if (i * 3 + j) % 2 == 0 else [0, 0, 0, 1]
            g.contype = g.conaffinity = 0
    # gravel base under the tiles (same scan texture at the same scale) so gaps between ragged tile edges show ground
    t = spec.add_texture()
    t.name, t.type, t.file = "rocks_base_tex", mujoco.mjtTexture.mjTEXTURE_2D, os.path.join(EXPORT, tex["a"])
    m = spec.add_material(); m.name = "rocks_base_mat"
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
    m.texuniform, m.texrepeat = True, [1 / (12.28 * ROCK_SCALE), 1 / (9.94 * ROCK_SCALE)]
    b = wb.add_geom()
    b.name, b.type, b.material = "rocks_base", mujoco.mjtGeom.mjGEOM_BOX, "rocks_base_mat"
    b.size = [(ROCKS_X[1] - ROCKS_X[0]) / 2 + 0.5, (ROCKS_Y[1] - ROCKS_Y[0]) / 2 + 0.5, 0.05]
    b.pos = [(ROCKS_X[0] + ROCKS_X[1]) / 2, (ROCKS_Y[0] + ROCKS_Y[1]) / 2, -0.04]
    b.contype = b.conaffinity = 0
