"""Convert assets/myscene.blend (exported headless by Blender to assets/myscene_export/raw/) into MuJoCo meshes.

Inputs:  raw/myscene.obj + scene_info.json (material -> texture), packed rock albedos, ship textures in
         assets/lcu1600class(1)/.
Outputs: assets/myscene_export/{<material>.obj, *.png, meta.json}
    - one OBJ per material (world coordinates, metres, z-up)
    - textures as PNG (MuJoCo loads PNG); the 16k rock albedos are downscaled to 2048 px
    - texture_props.jpg (dock) is missing from the download -> plain colour
"""
import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "assets", "myscene_export", "raw")
OUT = os.path.join(ROOT, "assets", "myscene_export")
SHIP_TEX = os.path.join(ROOT, "assets", "lcu1600class(1)")
MAX_TEX = 2048
Image.MAX_IMAGE_PIXELS = 16384 * 16384 + 1  # the rock albedos are 16k (known source); decoded at reduced size


def to_png(src, dst):
    if os.path.exists(dst):
        return
    im = Image.open(src)
    if im.format == "JPEG":
        im.draft("RGB", (MAX_TEX, MAX_TEX))  # decode huge JPEGs at reduced size
    im = im.convert("RGB")
    if max(im.size) > MAX_TEX:
        im.thumbnail((MAX_TEX, MAX_TEX))
    im.save(dst)


def write_obj(path, v, f, uv=None):
    with open(path, "w") as fo:
        fo.write("# myscene.blend export, metres, z-up\n")
        fo.write("".join(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in v))
        if uv is not None:
            fo.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in uv))
            fo.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f + 1))
        else:
            fo.write("".join(f"f {a} {b} {c}\n" for a, b, c in f + 1))


def main():
    info = json.load(open(os.path.join(RAW, "scene_info.json")))
    owner = {m: o for o, d in info["objects"].items() for m in d["materials"]}
    scene = trimesh.load(os.path.join(RAW, "myscene.obj"), force="scene", group_material=True)
    parts = {}
    for g in scene.geometry.values():
        parts.setdefault(getattr(g.visual.material, "name", "unknown"), []).append(g)
    meta = []
    for mname, gs in parts.items():
        m = trimesh.util.concatenate(gs) if len(gs) > 1 else gs[0]
        mi = info["materials"].get(mname, {"color": [0.6, 0.6, 0.6, 1], "texture": None})
        tex, src = None, None
        t = mi["texture"]
        if t and "packed" in t:
            src = os.path.join(RAW, t["packed"])
        elif t and "file" in t and os.path.exists(os.path.join(SHIP_TEX, t["file"])):
            src = os.path.join(SHIP_TEX, t["file"])
        uv = getattr(m.visual, "uv", None)
        if src and uv is not None:
            tex = os.path.splitext(os.path.basename(src))[0] + ".png"
            to_png(src, os.path.join(OUT, tex))
            rgba = [1, 1, 1, 1]
        else:
            uv = None
            rgba = [round(c, 3) for c in mi["color"][:3]] + [1.0]
            if t and "file" in t:  # texture referenced but not available (e.g. the dock's texture_props.jpg)
                rgba = [0.55, 0.52, 0.48, 1.0]
        fname = "ms_" + "".join(ch if ch.isalnum() else "_" for ch in mname.lower()) + ".obj"
        write_obj(os.path.join(OUT, fname), m.vertices, m.faces, uv)
        meta.append(dict(material=mname, object=owner.get(mname), obj=fname, texture=tex, rgba=rgba,
                         faces=int(len(m.faces)), lo=np.round(m.vertices.min(0), 3).tolist(),
                         hi=np.round(m.vertices.max(0), 3).tolist()))
        print(f"{mname:22s} {owner.get(mname, '?'):20s} faces {len(m.faces):7d} tex {tex}")
    json.dump(dict(source="assets/myscene.blend (user-provided)", meshes=meta), open(os.path.join(OUT, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
