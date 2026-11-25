"""MuJoCo version of assets/scene_terraine.blend: a 25 m x 5 m displaced floor (0-8 cm relief) for walking tests.

Collision + visual: heightfield sampled from the Blender-evaluated displaced surface (scripts/sample_scene_terrain.py,
2 cm cells), textured with the scene's albedo (one copy per 5 m tile, as in the .blend Array modifier).
A flat concrete apron at z = 0 surrounds it.
"""
import json
import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "assets", "scene_terrain")
ALBEDO_SRC = os.path.expanduser("~/Downloads/textures/text_Albedo.png")
APRON_TEX = os.path.join(ROOT, "assets", "dockyard", "textures", "concrete_floor_worn_001.png")


def _albedo():
    dst = os.path.join(D, "albedo_2k.png")
    if not os.path.exists(dst):
        from PIL import Image
        im = Image.open(ALBEDO_SRC).convert("RGB")
        im.thumbnail((2048, 2048))
        im.save(dst)
    return dst


def add_scene_terrain(spec):
    info = json.load(open(os.path.join(D, "terrain.json")))
    H = np.load(os.path.join(D, "heights.npy")).astype(float)
    wb = spec.worldbody
    for g in list(spec.geoms):
        if g.name == "floor":
            spec.delete(g)
    spec.visual.headlight.ambient = [0.4, 0.4, 0.42]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    sun = wb.add_light()
    sun.name, sun.pos, sun.dir, sun.type = "sun", [5, -10, 20], [0.2, 0.4, -0.9], mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse, sun.castshadow = [0.65, 0.63, 0.6], True

    def material(name, file, repeat, uniform=False):
        t = spec.add_texture()
        t.name, t.type, t.file = name + "_tex", mujoco.mjtTexture.mjTEXTURE_2D, file
        m = spec.add_material(); m.name = name
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
        m.texrepeat, m.texuniform = list(repeat), uniform
        return name

    apron = wb.add_geom()  # flat surround, slightly below the terrain's lowest point
    apron.name, apron.type, apron.size = "apron", mujoco.mjtGeom.mjGEOM_PLANE, [60, 60, 0.1]
    apron.pos, apron.material = [10, 0, -0.001], material("apron", APRON_TEX, (0.5, 0.5), uniform=True)

    zmin, zmax = float(H.min()), float(H.max())
    ny, nx = H.shape
    hf = spec.add_hfield()
    hf.name, hf.nrow, hf.ncol = "terrain_hf", ny, nx
    rx, ry = (info["x1"] - info["x0"]) / 2, (info["y1"] - info["y0"]) / 2
    hf.size = [rx, ry, zmax - zmin, 0.05]
    hf.userdata = ((H - zmin) / (zmax - zmin)).flatten().tolist()
    g = wb.add_geom()
    g.name, g.type, g.hfieldname = "terrain", mujoco.mjtGeom.mjGEOM_HFIELD, "terrain_hf"
    g.pos = [(info["x0"] + info["x1"]) / 2, (info["y0"] + info["y1"]) / 2, zmin]
    g.material, g.friction = material("terrain", _albedo(), (round(2 * rx / 5.0), round(2 * ry / 5.0))), [1.0, 0.02, 0.01]
    return dict(x0=info["x0"], x1=info["x1"], y0=info["y0"], y1=info["y1"], relief_m=zmax - zmin)
