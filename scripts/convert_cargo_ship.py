"""Convert the uploaded bulk-carrier Blender model (assets/cargo-ship-3d-model/cargo-ship2.blend) into MuJoCo meshes.

Step 1 (Blender, headless, --disable-autoexec so no embedded scripts run) exported it to OBJ + MTL and unpacked its
images into assets/dockyard/ship_cargo/raw/ - see docs/PROJECT_LOG.md. This script:
    - splits the OBJ by material, scales it x10 (model units -> a 138 m x 22 m Handysize bulk carrier)
    - centres it on x / y and puts the lowest hull point (keel) at z = 0; length along x, bow towards +x
    - colours from the MTL; hull split at the boot-top line: red topsides / dark red-brown anti-fouling bottom

    python scripts/convert_cargo_ship.py      # -> assets/dockyard/ship_cargo/{*.obj, hull_mask.png, meta.json}
"""
import json
import os

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "assets", "dockyard", "ship_cargo", "raw")
OUT = os.path.join(ROOT, "assets", "dockyard", "ship_cargo")
SCALE = 10.0
WATERLINE = 3.5                       # boot-top line above the keel [m] (model units x10)
ANTIFOUL = [0.32, 0.09, 0.07, 1.0]    # anti-fouling bottom paint


def write_obj(path, v, f, uv=None):
    with open(path, "w") as fo:
        fo.write("# bulk carrier (converted from cargo-ship2.blend), metres, z-up, keel at z=0\n")
        fo.write("".join(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in v))
        if uv is not None:
            fo.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in uv))
            fo.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f + 1))
        else:
            fo.write("".join(f"f {a} {b} {c}\n" for a, b, c in f + 1))


def main():
    scene = trimesh.load(os.path.join(RAW, "cargo_ship.obj"), force="scene", group_material=True)
    parts = {}
    for name, g in scene.geometry.items():
        mat = getattr(g.visual, "material", None)
        mname = getattr(mat, "name", name)
        parts.setdefault(mname, []).append(g)
    hull = trimesh.util.concatenate(parts["hull"])
    lo = hull.vertices.min(0) * SCALE
    hi = hull.vertices.max(0) * SCALE
    off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    meta = []
    for mname, gs in parts.items():
        m = trimesh.util.concatenate(gs) if len(gs) > 1 else gs[0]
        v = m.vertices * SCALE + off
        mat = getattr(m.visual, "material", None)
        kd = getattr(mat, "diffuse", None)
        rgba = (np.array(kd[:3]) / 255.0).round(3).tolist() + [1.0] if kd is not None else [0.6, 0.6, 0.6, 1.0]
        if mname == "hull":
            # The packed hullUV mask does not line up with these UVs (the 2013 file mapped it differently), so the hull
            # is painted by height instead: red topsides above the boot-top line, dark red-brown anti-fouling below.
            zc = v[m.faces].mean(1)[:, 2]
            for tag, sel, col in (("hull_topsides", zc >= WATERLINE, rgba), ("hull_bottom", zc < WATERLINE, ANTIFOUL)):
                fname = f"cargo_{tag}.obj"
                write_obj(os.path.join(OUT, fname), v, m.faces[sel])
                meta.append(dict(material=tag, obj=fname, rgba=col, faces=int(sel.sum()), texture=None))
                print(f"{tag:22s} faces {int(sel.sum()):6d} rgba {col}")
            continue
        fname = f"cargo_{mname.replace('.', '_').lower()}.obj"
        write_obj(os.path.join(OUT, fname), v, m.faces)
        meta.append(dict(material=mname, obj=fname, rgba=rgba, faces=int(len(m.faces)), texture=None))
        print(f"{mname:22s} faces {len(m.faces):6d} rgba {rgba} extent {np.round(v.max(0) - v.min(0), 1).tolist()}")
    ext = (hi - lo).round(2).tolist()
    json.dump(dict(source="cargo-ship2.blend (user-provided; licence not stated in the download)", scale=SCALE,
                   hull_extent_m=ext, meshes=meta), open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print("hull extent (L, B, H) m:", ext)


if __name__ == "__main__":
    main()
