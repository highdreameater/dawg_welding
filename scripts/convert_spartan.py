"""Convert the MV Spartan glTF (Scottish Maritime Museum, Sketchfab, CC0) into MuJoCo-loadable OBJ meshes + PNG
textures for the dockyard scene.

    python scripts/convert_spartan.py      # assets/mv-spartan-3d-model/ -> assets/dockyard/ship_spartan/

- node transforms baked in; glTF Y-up -> MuJoCo Z-up (x, y, z) -> (x, -z, y); length along x, beam along y
- centred on x / y, keel (lowest point) at z = 0, units already metres
- the material -> texture link is missing in the export; textures are matched by name (<material>_diffuse.png)
- 'Unwrap Texture Checker' box is a leftover placeholder and is skipped
"""
import json
import os
import shutil

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "mv-spartan-3d-model")
OUT = os.path.join(ROOT, "assets", "dockyard", "ship_spartan")


def write_obj(path, v, f, uv, n):
    with open(path, "w") as fo:
        fo.write("# MV Spartan (Scottish Maritime Museum, CC0) - converted from glTF, Z-up, metres\n")
        fo.write("".join(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in v))
        fo.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in uv))
        fo.write("".join(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in n))
        fo.write("".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in f + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(SRC, "scene.gltf")))["asset"].get("extras", {})
    scene = trimesh.load(os.path.join(SRC, "scene.gltf"), force="scene")
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # y-up -> z-up
    keep = [p for p in parts if "Unwrap" not in (getattr(p.visual.material, "name", "") or "")]
    allv = np.vstack([p.vertices @ R.T for p in keep])
    lo, hi = allv.min(0), allv.max(0)
    off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    out = []
    for i, p in enumerate(keep):
        mat = p.visual.material.name
        v = p.vertices @ R.T + off
        n = p.vertex_normals @ R.T
        uv = p.visual.uv if getattr(p.visual, "uv", None) is not None else np.zeros((len(v), 2))
        name = f"spartan_{i:02d}_{'hull' if mat == 'default' else mat.split('_')[0].lower()}"
        write_obj(os.path.join(OUT, name + ".obj"), v, p.faces, uv, n)
        tex_src = os.path.join(SRC, "textures", f"{mat}_diffuse.png")
        tex = None
        if os.path.exists(tex_src):
            tex = f"{mat}_diffuse.png"
            if not os.path.exists(os.path.join(OUT, tex)):
                from PIL import Image
                Image.open(tex_src).convert("RGB").save(os.path.join(OUT, tex))  # 1-bit palette PNGs -> RGB
        out.append(dict(name=name, obj=name + ".obj", texture=tex, material=mat, faces=int(len(p.faces)),
                        is_hull=mat == "default"))
        print(name, len(p.faces), "faces", np.round(v.max(0) - v.min(0), 2).tolist(), tex)
    ext = (hi - lo).round(3).tolist()
    json.dump(dict(source=meta, extent_m=ext, meshes=out), open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print("extent (L, B, H) m:", ext)


if __name__ == "__main__":
    main()
