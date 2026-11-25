"""Blender (headless) step: evaluate assets/scene_terraine.blend's displaced floor and sample it into a heightfield.

    /Applications/Blender.app/Contents/MacOS/Blender -b assets/scene_terraine.blend --disable-autoexec \
        --python scripts/sample_scene_terrain.py

The floor is a plane with Subdivision (simple) + 2 x Array + Displace (8 cm, image texture via UV). Fully evaluated it is
125 M triangles, so the subdivision is lowered (displacement detail ~2 cm) before evaluating; the world-space vertices
are binned onto a 2 cm grid taking the top surface (max z) per cell, gaps filled from neighbours.
Output: assets/scene_terrain/heights.npy (rows = y, cols = x, metres) + terrain.json (bounds, cell).
"""
import json
import os

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "scene_terrain")
CELL = 0.02
os.makedirs(OUT, exist_ok=True)

ob = bpy.data.objects["floor_LOD0_floor"]
for m in ob.modifiers:
    if m.type == "SUBSURF":
        m.levels = 1  # base 129x129 grid per 5 m tile (3.9 cm) -> 1.95 cm after one simple subdivision
dg = bpy.context.evaluated_depsgraph_get()
ev = ob.evaluated_get(dg)
me = ev.to_mesh()
n = len(me.vertices)
co = np.empty(n * 3, dtype=np.float32)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
M = np.array(ob.matrix_world)
co = co @ M[:3, :3].T + M[:3, 3]
ev.to_mesh_clear()
print("evaluated vertices", n, "world bounds", co.min(0).round(3), co.max(0).round(3))

lo, hi = co.min(0), co.max(0)
nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1, int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
ix = np.clip(((co[:, 0] - lo[0]) / CELL).round().astype(int), 0, nx - 1)
iy = np.clip(((co[:, 1] - lo[1]) / CELL).round().astype(int), 0, ny - 1)
H = np.full((ny, nx), -np.inf)
np.maximum.at(H, (iy, ix), co[:, 2])
miss = ~np.isfinite(H)
for _ in range(20):  # fill empty cells from neighbours
    if not miss.any():
        break
    P = np.pad(H, 1, constant_values=-np.inf)
    nb = np.max(np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]]), 0)
    H[miss] = nb[miss]
    miss = ~np.isfinite(H)
H[~np.isfinite(H)] = float(np.nanmin(H[np.isfinite(H)]))
np.save(os.path.join(OUT, "heights.npy"), H.astype(np.float32))
json.dump(dict(x0=float(lo[0]), x1=float(lo[0] + (nx - 1) * CELL), y0=float(lo[1]), y1=float(lo[1] + (ny - 1) * CELL),
               cell=CELL, z_min=float(H.min()), z_max=float(H.max()), shape=[ny, nx], filled_cells=int((~np.isfinite(H)).sum())),
          open(os.path.join(OUT, "terrain.json"), "w"), indent=2)
print("heightfield", H.shape, "z", round(float(H.min()), 4), "..", round(float(H.max()), 4))
