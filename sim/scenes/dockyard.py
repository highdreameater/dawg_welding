"""Dry dockyard scene for MuJoCo: a ship ashore on its keel blocks, a short stair (8 steps) up to a scaffold platform
alongside the lower hull, textured bumpy concrete yard, stones, crates and barrels. No water.

Ship options:
    "spartan"    (default) MV Spartan - Clyde puffer, photogrammetry by the Scottish Maritime Museum (Sketchfab, CC0),
                 converted by scripts/convert_spartan.py -> assets/dockyard/ship_spartan/. Includes its concrete plinth
                 and timber keel chocks. Visual only (the robot's lane never touches it).
    "procedural" generated 60 m steel cargo ship with a curved bilge (fallback).
Other downloaded assets (Poly Haven, CC0): textures + stones/crates/barrels, see assets/dockyard/CREDITS.md.

Layout (robot starts at the origin facing +x; the ship is on its left, +y):
    yard     : x in [-30, 40], y in [-10, 30], bumpy concrete heightfield (0..3 cm), friction 1.0
    stairs   : 2 flights x 4 steps (10 cm rise, 30 cm run) + 1.2 m landing, lane y in [3.6, 4.8], from x = 3 m
    scaffold : 0.8 m high work platform continuing along +x beside the lower hull
    hull     : ~0.6 m left of the lane centre at ~0.6 m above the platform (inside the Z1's sideways reach band)

    from scenes.dockyard import add_dockyard;  build_model(cfg, ..., scene_fn=add_dockyard)
"""
import json
import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "assets", "dockyard")
GEN = os.path.join(ASSETS, "generated")

YARD_X0, YARD_X1, YARD_Y0, YARD_Y1 = -70.0, 100.0, -25.0, 35.0  # fits the 137 m ship
CELL, BUMP = 0.25, 0.03                       # heightfield cell [m], relief [m]
STAIR_X0, STAIR_Y0, STAIR_Y1 = 3.0, 3.6, 4.8  # 1.2 m wide lane
RISE, RUN, NSTEP, NFLIGHT, LANDING = 0.10, 0.30, 4, 2, 1.2
WORK_Z = RISE * NSTEP * NFLIGHT               # scaffold platform height 0.8 m
LANE_Y = (STAIR_Y0 + STAIR_Y1) / 2
WELD_DZ, WELD_GAP = 0.6, 0.6                  # hull surface at platform+0.6 m sits 0.6 m left of the lane centre

# MV Spartan, ship frame: length along x (+-11.5 m), plinth bottom z = 0, centreline y = 0. Measured from the scan:
# plinth 0..~0.25 m, timber chocks up to 0.70 m (out to |y| 3.55), hull from 0.75 m,
# door-side (-y) hull surface at y = -2.53 between 1.0 and 2.0 m (nearly vertical, curving in towards the keel).
SPARTAN_DIR = os.path.join(ASSETS, "ship_spartan")
SPARTAN_SIDE_Y, SPARTAN_HALF_L = -2.53, 11.5
SPARTAN_SCAFFOLD_LEN = 16.0                   # stairs start off the stern end of the plinth; scaffold runs amidships

# Bulk carrier (user-provided cargo-ship2.blend, converted by scripts/convert_cargo_ship.py, x10 -> 137 x 22 x 14 m).
# Ship frame: bow +x, keel at z = 0, centreline y = 0. Rounded V-ish lower hull, midship half-beam vs height above
# keel: 0.25 m -> 1.29, 0.75 m -> 2.41, 1.5 m -> 3.29, 2.5 m -> 5.63, 4 m -> 8.2 (overhangs the work lane).
CARGO_DIR = os.path.join(ASSETS, "ship_cargo")
CARGO_BLOCK_H = 1.0           # keel blocks: keel at 1.0 m
CARGO_LANE_OFF = 2.62         # lane centre |y| from the keel line: hull 0.55 m left at z 1.6 m, 0.48 m above the robot
CARGO_SCAFFOLD_LEN = 16.0

# procedural fallback ship
SHIP_L, SHIP_B, KEEL_Z, DECK_Z = 60.0, 14.0, 0.6, 10.6
PROC_SCAFFOLD_LEN = 8.0


def hull_half_beam(z, fx=1.0):
    """Procedural hull half-beam at height z amidships: rounded bilge, ~8 % flare towards the deck."""
    u = np.clip((z - KEEL_Z) / (DECK_Z - KEEL_Z), 0, 1)
    return SHIP_B / 2 * fx * np.clip(u / 0.22, 0, 1) ** 0.5 * (0.92 + 0.08 * u)


def stair_segments():
    segs, x, z = [], STAIR_X0, 0.0
    for f in range(NFLIGHT):
        segs.append(("flight", x, z))
        x += NSTEP * RUN
        z += NSTEP * RISE
        if f < NFLIGHT - 1:
            segs.append(("landing", x, z))
            x += LANDING
    return segs, x  # x = start of the scaffold platform


# ----------------------------------------------------------------------------------------------- generated meshes
def _write_obj(path, V, F, UV):
    with open(path, "w") as f:
        f.write("# procedural (dockyard.py)\n")
        f.write("".join(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n" for v in V))
        f.write("".join(f"vt {t[0]:.4f} {t[1]:.4f}\n" for t in UV))
        f.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in np.asarray(F) + 1))


def make_hull(path_hull, path_deck, nx=90, nz=40):
    s = np.linspace(-1, 1, nx)
    fx = np.where(s > 0, (1 - np.abs(s) ** 2.4) ** 0.5, (1 - np.abs(s) ** 7) ** 0.35)
    zs = KEEL_Z + np.linspace(0, 1, nz) ** 1.5 * (DECK_Z - KEEL_Z)
    V, UV, F = [], [], []
    idx = np.zeros((2, nx, nz), int)
    for side, sg in enumerate((-1, 1)):
        for i in range(nx):
            for j in range(nz):
                x, y = s[i] * SHIP_L / 2, sg * hull_half_beam(zs[j], fx[i])
                idx[side, i, j] = len(V)
                V.append((x, y, zs[j] - DECK_Z)); UV.append((x / 3.0, zs[j] / 3.0))
    for side in (0, 1):
        for i in range(nx - 1):
            for j in range(nz - 1):
                a, b, c, d = idx[side, i, j], idx[side, i + 1, j], idx[side, i + 1, j + 1], idx[side, i, j + 1]
                F += [(a, b, c), (a, c, d)] if side == 1 else [(a, c, b), (a, d, c)]
    _write_obj(path_hull, V, F, UV)
    Vd, UVd, Fd = [], [], []
    for i in range(nx):
        for sg in (-1, 1):
            x, y = s[i] * SHIP_L / 2, sg * hull_half_beam(DECK_Z, fx[i])
            Vd.append((x, y, 0.0)); UVd.append((x / 2.0, y / 2.0))
    for i in range(nx - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        Fd += [(a, d, c), (a, c, b)]
    _write_obj(path_deck, Vd, Fd, UVd)


def yard_heights(nrow, ncol, seed=7):
    """Smooth, low relief (a few cm): worn concrete, not off-road."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:nrow, 0:ncol]
    h = np.zeros((nrow, ncol))
    for _ in range(14):
        fxk, fyk = rng.uniform(0.02, 0.18, 2)
        h += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (fxk * xx + fyk * yy) / 7 + rng.uniform(0, 2 * np.pi))
    h += 0.25 * rng.standard_normal((nrow, ncol))
    k = np.ones(5) / 5
    h = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, h)
    h = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, h)
    return (h - h.min()) / (h.max() - h.min())


# ----------------------------------------------------------------------------------------------- spec helpers
def _material(spec, name, tex_file, repeat=(1, 1), rgba=(1, 1, 1, 1), uniform=False):
    t = spec.add_texture()
    t.name, t.type, t.file = name + "_tex", mujoco.mjtTexture.mjTEXTURE_2D, tex_file
    m = spec.add_material()
    m.name = name
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
    m.texrepeat, m.texuniform, m.rgba = list(repeat), uniform, list(rgba)
    return name


def _box(body, pos, half, material=None, rgba=None, friction=None, contact=True):
    g = body.add_geom()
    g.type, g.pos, g.size = mujoco.mjtGeom.mjGEOM_BOX, list(pos), list(half)
    if material:
        g.material = material
    if rgba is not None:
        g.rgba = list(rgba)
    if friction is not None:
        g.friction = [friction, 0.02, 0.01]
    if not contact:
        g.contype = g.conaffinity = 0
    return g


def _post(body, x, y, z_top, rgba=(0.25, 0.27, 0.3, 1)):
    p = body.add_geom()
    p.type, p.size, p.pos, p.rgba = mujoco.mjtGeom.mjGEOM_CYLINDER, [0.04, z_top / 2, 0], [x, y, z_top / 2], list(rgba)


def _rail(body, a, b, rgba=(0.9, 0.75, 0.1, 1)):
    r = body.add_geom()
    r.type, r.size, r.fromto = mujoco.mjtGeom.mjGEOM_CAPSULE, [0.025, 0, 0], list(a) + list(b)
    r.rgba, r.contype, r.conaffinity = list(rgba), 0, 0


def _obj_extent(path):
    v = np.array([list(map(float, l.split()[1:4])) for l in open(path) if l.startswith("v ")])
    return v.min(0), v.max(0)


# ----------------------------------------------------------------------------------------------- scene parts
def _add_spartan(spec, wb, x_plat):
    meta = json.load(open(os.path.join(SPARTAN_DIR, "meta.json")))
    ship_x = x_plat + SPARTAN_HALF_L + 0.5          # stairs end just past the stern end of the plinth
    ship_y = LANE_Y + WELD_GAP - SPARTAN_SIDE_Y     # hull 0.6 m left of the lane centre at arm height
    sb = wb.add_body(); sb.name, sb.pos = "ship", [ship_x, ship_y, 0.01]
    made = set()
    for mm in meta["meshes"]:
        me = spec.add_mesh(); me.name, me.file = mm["name"], os.path.join(SPARTAN_DIR, mm["obj"])
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        gs = sb.add_geom()
        gs.name, gs.type, gs.meshname = mm["name"], mujoco.mjtGeom.mjGEOM_MESH, mm["name"]
        gs.contype = gs.conaffinity = 0
        if mm["texture"]:
            mat = "mat_spartan_" + mm["material"]
            if mat not in made:
                _material(spec, mat, os.path.join(SPARTAN_DIR, mm["texture"]))
                made.add(mat)
            gs.material = mat
    return ship_x, ship_y, SPARTAN_SCAFFOLD_LEN


def _add_cargo(spec, wb, x_plat, m_conc):
    meta = json.load(open(os.path.join(CARGO_DIR, "meta.json")))
    ship_x = x_plat + CARGO_SCAFFOLD_LEN / 2       # scaffold alongside the parallel midbody
    ship_y = LANE_Y + CARGO_LANE_OFF
    sb = wb.add_body(); sb.name, sb.pos = "ship", [ship_x, ship_y, CARGO_BLOCK_H]
    for mm in meta["meshes"]:
        name = "cargo_" + mm["material"].replace(".", "_")
        me = spec.add_mesh(); me.name, me.file = name, os.path.join(CARGO_DIR, mm["obj"])
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        gs = sb.add_geom()
        gs.name, gs.type, gs.meshname = name, mujoco.mjtGeom.mjGEOM_MESH, name
        gs.contype = gs.conaffinity = 0                # visual only: the lane never touches the ship
        if mm["texture"]:                              # hull: red x (white topsides / black anti-fouling) mask
            gs.material = _material(spec, "mat_cargo_hull", os.path.join(CARGO_DIR, mm["texture"]), rgba=mm["rgba"])
        else:
            gs.rgba = mm["rgba"]
    for i, lx in enumerate(np.arange(x_plat - 2.0, x_plat + CARGO_SCAFFOLD_LEN + 0.1, 4.0)):
        # dry-dock work floodlights on the outer side of the lane, aimed up/in at the hull bottom
        lt = wb.add_light()
        lt.name, lt.pos, lt.dir = f"worklight_{i}", [lx, STAIR_Y0 - 0.4, 0.4], [0, 0.55, 0.83]
        lt.diffuse, lt.cutoff, lt.exponent, lt.castshadow = [0.45, 0.42, 0.38], 75, 2, False
        lt.attenuation = [1.0, 0.05, 0.01]
    for kx in np.arange(-60.0, 61.0, 4.0):            # keel blocks (concrete) with timber caps
        _box(wb, [ship_x + kx, ship_y, CARGO_BLOCK_H / 2 - 0.08], [0.5, 0.8, CARGO_BLOCK_H / 2 - 0.08], m_conc,
             rgba=[0.8, 0.8, 0.8, 1])
        _box(wb, [ship_x + kx, ship_y, CARGO_BLOCK_H - 0.08], [0.55, 0.85, 0.08], rgba=[0.45, 0.32, 0.2, 1])
    return ship_x, ship_y, CARGO_SCAFFOLD_LEN


def _add_procedural_ship(spec, wb, m_hull, m_deck, m_iron, m_conc):
    ship_y = LANE_Y + WELD_GAP + hull_half_beam(WORK_Z + WELD_DZ)
    os.makedirs(GEN, exist_ok=True)
    hull_obj, deck_obj = os.path.join(GEN, "ship_hull.obj"), os.path.join(GEN, "ship_deck.obj")
    make_hull(hull_obj, deck_obj)
    for name, f in (("ship_hull", hull_obj), ("ship_deck", deck_obj)):
        me = spec.add_mesh(); me.name, me.file = name, f
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    ship = wb.add_body(); ship.name, ship.pos = "ship", [0, ship_y, DECK_Z]
    gh = ship.add_geom()
    gh.name, gh.type, gh.meshname, gh.material = "hull", mujoco.mjtGeom.mjGEOM_MESH, "ship_hull", m_hull
    gd = ship.add_geom()
    gd.type, gd.meshname, gd.material = mujoco.mjtGeom.mjGEOM_MESH, "ship_deck", m_deck
    gd.pos, gd.contype, gd.conaffinity = [0, 0, 0.004], 0, 0
    _box(ship, [-22, 0, 4.0], [5, 5.5, 4.0], m_iron, rgba=[0.9, 0.9, 0.92, 1])
    _box(ship, [-22, 0, 8.6], [3, 4.0, 0.6], rgba=[0.15, 0.2, 0.25, 1])
    _box(ship, [-24, 0, 11], [0.6, 0.6, 2.0], rgba=[0.75, 0.15, 0.1, 1])
    cols = [[0.75, 0.2, 0.15, 1], [0.15, 0.35, 0.65, 1], [0.2, 0.55, 0.3, 1], [0.85, 0.6, 0.15, 1]]
    for k, (cx, cy) in enumerate([(cx, cy) for cx in (-6, 2, 10, 18) for cy in (-3.2, 0.0, 3.2)]):
        _box(ship, [cx, cy, 1.3], [3.05, 1.22, 1.3], m_iron, rgba=cols[k % 4])
    for kx in np.arange(-24, 25, 4.0):
        _box(wb, [kx, ship_y, KEEL_Z / 2], [0.5, 1.0, KEEL_Z / 2], m_conc, rgba=[0.75, 0.75, 0.75, 1])
    return 0.0, ship_y, PROC_SCAFFOLD_LEN


def _add_stairs(wb, segs, x_plat, scaffold_len, m_grate):
    """8-step stair (2 x 4, closed risers/sides) + landing + scaffold; handrail on the outer (right) side only."""
    hw = (STAIR_Y1 - STAIR_Y0) / 2
    st = wb.add_body(); st.name = "stairs"
    for kind, x0, z0 in segs:
        if kind == "flight":
            for i in range(NSTEP):
                zt = z0 + RISE * (i + 1)
                _box(st, [x0 + RUN * i + RUN / 2, LANE_Y, zt - 0.02], [RUN / 2, hw, 0.02], m_grate, friction=0.9)
                _box(st, [x0 + RUN * i + 0.01, LANE_Y, zt - RISE / 2 - 0.02], [0.01, hw, RISE / 2], m_grate)
                for ys in (STAIR_Y0, STAIR_Y1):
                    _box(st, [x0 + RUN * i + RUN / 2, ys, zt / 2], [RUN / 2, 0.02, zt / 2], rgba=[0.25, 0.27, 0.3, 1])
        else:
            _box(st, [x0 + LANDING / 2, LANE_Y, z0 - 0.03], [LANDING / 2, hw, 0.03], m_grate, friction=0.9)
            for xp in (x0 + 0.1, x0 + LANDING - 0.1):
                for yp in (STAIR_Y0 + 0.06, STAIR_Y1 - 0.06):
                    _post(st, xp, yp, z0 - 0.06)
    x_end = x_plat + scaffold_len
    _box(st, [(x_plat + x_end) / 2, LANE_Y, WORK_Z - 0.03], [scaffold_len / 2, hw, 0.03], m_grate, friction=0.9)
    for xp in np.arange(x_plat + 0.1, x_end, 2.0):
        for yp in (STAIR_Y0 + 0.06, STAIR_Y1 - 0.06):
            _post(st, xp, yp, WORK_Z - 0.06)
    yr = STAIR_Y0 - 0.04
    f1_end, f2_start = segs[0][1] + NSTEP * RUN, segs[2][1]
    for a, b in [((STAIR_X0, 0.9), (f1_end, NSTEP * RISE + 0.9)), ((f1_end, NSTEP * RISE + 0.9), (f2_start, NSTEP * RISE + 0.9)),
                 ((f2_start, NSTEP * RISE + 0.9), (x_plat, WORK_Z + 0.9)), ((x_plat, WORK_Z + 0.9), (x_end, WORK_Z + 0.9))]:
        _rail(st, [a[0], yr, a[1]], [b[0], yr, b[1]])
    return x_end


def _add_clutter(spec, wb, man):
    """Stones, crates, barrels (Poly Haven; glTF is y-up -> rotate +90 deg about x), clear of the lane and the ship."""
    placements = [
        ("boulder_01", -6, -4, 1.4, 0.3), ("rock_07", -3, 1.0, 0.6, 1.2), ("rock_09", 6, -3.0, 0.8, 2.0),
        ("namaqualand_boulder_02", 13, -6, 1.6, 0.7), ("boulder_01", -12, 1.5, 1.2, 2.6), ("stone_01", 1.5, -2.0, 0.35, 0.4),
        ("rock_07", 9, 0.5, 0.45, 2.2), ("stone_01", -1.5, -5.5, 0.3, 1.0), ("rock_09", -16, -6, 1.0, 0.1),
        ("boulder_01", 24, -2, 1.1, 1.9), ("rock_09", 33, 1.5, 0.9, 0.8),
        ("wooden_crate_01", -8, -1.5, 0.9, 0.2), ("wooden_crate_01", -9.1, -1.3, 0.9, 1.4), ("barrel_03", -10.3, -1.0, 0.9, 0.0),
        ("barrel_03", -10.9, -1.7, 0.9, 0.0), ("wooden_crate_01", 18, -4.5, 0.9, 0.5),
    ]
    added = set()
    for i, (mid, x, y, size, yaw) in enumerate(placements):
        y -= 9.0  # keep the yard clutter out from under the (overhanging) hull
        if mid not in man["models"]:
            continue
        info = man["models"][mid]
        obj = os.path.join(ASSETS, info["obj"])
        lo, hi = _obj_extent(obj)
        sc = size / float((hi - lo).max())
        if mid not in added:
            me = spec.add_mesh(); me.name, me.file, me.scale = mid, obj, [sc, sc, sc]
            if info.get("texture"):
                _material(spec, "mat_" + mid, os.path.join(ASSETS, info["texture"]))
            added.add(mid)
        b = wb.add_body(); b.name = f"{mid}_{i}"
        b.pos = [x, y, -lo[1] * sc - 0.01]
        b.alt.euler = [np.pi / 2, yaw, 0]
        gm = b.add_geom()
        gm.type, gm.meshname, gm.friction = mujoco.mjtGeom.mjGEOM_MESH, mid, [0.9, 0.02, 0.01]
        if info.get("texture"):
            gm.material = "mat_" + mid


def add_dockyard(spec, ship="cargo"):
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    T = lambda tid: os.path.join(ASSETS, man["textures"][tid]["file"])
    wb = spec.worldbody

    for g in list(spec.geoms):  # drop the robot scene's flat floor
        if g.name == "floor":
            spec.delete(g)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.rgba.haze = [0.72, 0.78, 0.84, 1]
    spec.visual.map.zfar = 300
    sun = wb.add_light()
    sun.name, sun.pos, sun.dir, sun.type = "sun", [0, -20, 40], [0.3, 0.5, -0.8], mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse, sun.castshadow = [0.7, 0.68, 0.64], True

    m_yard = _material(spec, "yard", T("concrete_floor_worn_001"), repeat=((YARD_X1 - YARD_X0) / 2, (YARD_Y1 - YARD_Y0) / 2))
    m_grate = _material(spec, "grate", T("metal_grate_rusty"), repeat=(2, 2), uniform=True)

    # --- dry yard: bumpy concrete heightfield, friction 1.0
    nrow, ncol = int((YARD_Y1 - YARD_Y0) / CELL) + 1, int((YARD_X1 - YARD_X0) / CELL) + 1
    hf = spec.add_hfield()
    hf.name, hf.nrow, hf.ncol = "yard_hf", nrow, ncol
    hf.size = [(YARD_X1 - YARD_X0) / 2, (YARD_Y1 - YARD_Y0) / 2, BUMP, 0.5]
    hf.userdata = yard_heights(nrow, ncol).flatten().tolist()
    g = wb.add_geom()
    g.name, g.type, g.hfieldname = "yard", mujoco.mjtGeom.mjGEOM_HFIELD, "yard_hf"
    g.pos, g.material, g.friction = [(YARD_X0 + YARD_X1) / 2, (YARD_Y0 + YARD_Y1) / 2, 0], m_yard, [1.0, 0.02, 0.01]

    segs, x_plat = stair_segments()
    m_conc = _material(spec, "blocks", T("painted_concrete"), uniform=True)
    if ship == "cargo":
        ship_x, ship_y, scaffold_len = _add_cargo(spec, wb, x_plat, m_conc)
    elif ship == "spartan":
        ship_x, ship_y, scaffold_len = _add_spartan(spec, wb, x_plat)
    else:
        m_hull = _material(spec, "hull", T("rusty_painted_metal"))
        m_deck = _material(spec, "deck", T("metal_plate_02"))
        m_iron = _material(spec, "iron", T("rusty_corrugated_iron"), uniform=True)
        ship_x, ship_y, scaffold_len = _add_procedural_ship(spec, wb, m_hull, m_deck, m_iron, m_conc)
    x_end = _add_stairs(wb, segs, x_plat, scaffold_len, m_grate)
    _add_clutter(spec, wb, man)
    return dict(scaffold_x0=x_plat, scaffold_x1=x_end, work_z=WORK_Z, lane_y=LANE_Y, ship_x=ship_x, ship_y=ship_y)
