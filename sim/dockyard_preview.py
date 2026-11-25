"""Render the dockyard scene on its own (no robot): overview stills + a slow orbit video.

    python sim/dockyard_preview.py            # -> results/dockyard/{overview,stairs_hull,under_bilge}.png, orbit.mp4

To use the scene with the robot: build_model(cfg, ..., scene_fn=add_dockyard) (see sim/scenes/dockyard.py).
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scenes.dockyard import ROOT, add_dockyard  # noqa: E402

OUT = os.path.join(ROOT, "results", "dockyard")


def build():
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 900
    sky = spec.add_texture()
    sky.name, sky.type, sky.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2, sky.width, sky.height = [0.55, 0.7, 0.85], [0.15, 0.2, 0.3], 512, 3072
    info = add_dockyard(spec)
    return spec.compile(), info


def main():
    import imageio
    os.makedirs(OUT, exist_ok=True)
    m, info = build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print({k: round(float(v), 3) for k, v in info.items()}, "| geoms", m.ngeom, "meshes", m.nmesh)
    ren = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    # camera sits at lookat - distance * (cos az, sin az): positive azimuth = yard side (-y), facing the ship
    shots = {"overview": ([14, 7, 6], 120, 50, -18), "stairs_hull": ([6, 4.4, 1.2], 9, 55, -8),
             "under_bilge": ([16, 5.0, 1.5], 4.5, 115, 4)}
    for name, (look, dist, az, el) in shots.items():
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
        ren.update_scene(d, camera=cam)
        imageio.imwrite(os.path.join(OUT, f"{name}.png"), ren.render())
    frames = []
    for i in range(300):  # 10 s orbit around the stair / scaffold area
        cam.lookat[:], cam.distance, cam.elevation = [14, 6, 4.0], 45, -15
        cam.azimuth = 15 + 130 * i / 299
        ren.update_scene(d, camera=cam)
        frames.append(ren.render())
    imageio.mimsave(os.path.join(OUT, "orbit.mp4"), frames, fps=30, quality=7)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
