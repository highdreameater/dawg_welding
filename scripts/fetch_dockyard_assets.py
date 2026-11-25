"""Download the CC0 Poly Haven assets used by the dockyard scene (textures + rock/crate models) and convert models
to MuJoCo-loadable OBJ + PNG.

    python scripts/fetch_dockyard_assets.py          # -> assets/dockyard/{textures,models}/, CREDITS.md

All assets: Poly Haven (https://polyhaven.com), license CC0. Downloaded files are data only (images, glTF/bin).
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "dockyard")
API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "dockyard-scene-fetch/1.0"}

# texture id -> role in the scene
TEXTURES = {
    "concrete_floor_worn_001": "quay ground",
    "rusty_painted_metal": "ship hull",
    "metal_plate_02": "ship deck / gangway",
    "metal_grate_rusty": "stair treads / landings",
    "painted_concrete": "quay wall",
    "rusty_corrugated_iron": "containers / deck house",
}
# model id -> role
MODELS = {
    "boulder_01": "stone", "rock_07": "stone", "rock_09": "stone", "namaqualand_boulder_02": "stone",
    "stone_01": "stone",
    "wooden_crate_01": "dock clutter", "barrel_03": "dock clutter",
}


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        f.write(r.read())
    os.replace(path + ".part", path)


def files(asset_id):
    req = urllib.request.Request(API + asset_id, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_texture(tid):
    info = files(tid)
    diff = info.get("Diffuse") or info.get("diffuse")
    res = "1k" if "1k" in diff else sorted(diff)[0]
    path = os.path.join(OUT, "textures", f"{tid}.png")  # MuJoCo loads PNG
    if os.path.exists(path):
        return path
    if "jpg" in diff[res]:  # JPG is several times smaller to download; convert locally
        from PIL import Image
        jpg = os.path.join(OUT, "raw", "textures", f"{tid}.jpg")
        get(diff[res]["jpg"]["url"], jpg)
        Image.open(jpg).convert("RGB").save(path)
    else:
        get(diff[res]["png"]["url"], path)
    return path


def fetch_model(mid):
    info = files(mid)
    g = info["gltf"]["1k"]["gltf"]
    d = os.path.join(OUT, "raw", mid)  # each download in its own directory
    get(g["url"], os.path.join(d, os.path.basename(g["url"])))
    for rel, inc in g.get("include", {}).items():
        if rel.endswith(".bin") or "_diff_" in rel:  # geometry + colour map only (MuJoCo ignores normal/ARM maps)
            get(inc["url"], os.path.join(d, rel))
        else:  # placeholder so the glTF still resolves all its image references
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            p = os.path.join(d, rel)
            if not os.path.exists(p):
                from PIL import Image
                Image.new("RGB", (4, 4), (128, 128, 255)).save(p, format="JPEG")
    return os.path.join(d, os.path.basename(g["url"]))


def convert(mid, gltf_path):
    """glTF -> single OBJ (positions, UVs, normals) + its base-color texture as PNG."""
    import numpy as np
    import trimesh
    scene = trimesh.load(gltf_path, force="scene")
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
    m = max(meshes, key=lambda x: len(x.faces))  # the main mesh
    out_obj = os.path.join(OUT, "models", f"{mid}.obj")
    os.makedirs(os.path.dirname(out_obj), exist_ok=True)
    uv = getattr(m.visual, "uv", None)
    v, f, n = m.vertices, m.faces, m.vertex_normals
    with open(out_obj, "w") as fo:
        fo.write(f"# {mid} (Poly Haven, CC0) - converted from glTF\n")
        for p in v:
            fo.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        if uv is not None:
            for t in uv:
                fo.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
        for q in n:
            fo.write(f"vn {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}\n")
        for a, b, c in f + 1:
            fo.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" if uv is not None else f"f {a}//{a} {b}//{b} {c}//{c}\n")
    tex = None
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if img is None and mat is not None:
        img = getattr(mat, "image", None)
    if img is not None:
        tex = os.path.join(OUT, "models", f"{mid}.png")
        img.convert("RGB").save(tex)
    ext = (v.max(0) - v.min(0)).round(3).tolist()
    return dict(obj=os.path.relpath(out_obj, OUT), texture=os.path.relpath(tex, OUT) if tex else None,
                faces=int(len(f)), extent_m=ext, up_axis="y (glTF)")


def main():
    manifest = dict(textures={}, models={})
    for tid, role in TEXTURES.items():
        try:
            manifest["textures"][tid] = dict(file=os.path.relpath(fetch_texture(tid), OUT), role=role)
            print("texture ok", tid)
        except Exception as e:
            print("texture FAILED", tid, e, file=sys.stderr)
    for mid, role in MODELS.items():
        try:
            info = convert(mid, fetch_model(mid))
            manifest["models"][mid] = dict(role=role, **info)
            print("model ok", mid, info["faces"], "faces", info["extent_m"])
        except Exception as e:
            print("model FAILED", mid, e, file=sys.stderr)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT, "CREDITS.md"), "w") as f:
        f.write("# Dockyard scene assets\n\nAll downloaded assets are from [Poly Haven](https://polyhaven.com), license **CC0** "
                "(public domain). Models were converted from glTF (1k) to OBJ + PNG by `scripts/fetch_dockyard_assets.py`.\n\n")
        f.write("| asset | type | used as |\n|---|---|---|\n")
        for k, v in manifest["textures"].items():
            f.write(f"| [{k}](https://polyhaven.com/a/{k}) | texture | {v['role']} |\n")
        for k, v in manifest["models"].items():
            f.write(f"| [{k}](https://polyhaven.com/a/{k}) | model | {v['role']} |\n")
        f.write("\nMade procedurally in `sim/scenes/dockyard.py` (no free modern steel ship model was found): ship hull "
                "(curved bilge, flared steel sides), deck, deck house, deck containers, keel blocks, 8-step stair + "
                "scaffold platform, bumpy yard heightfield. Textures on these are the Poly Haven ones above.\n")


if __name__ == "__main__":
    main()
