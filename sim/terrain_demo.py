"""Go2 + Z1 (arm folded) walking the full length of the scene_terraine.blend floor (25 m, 0-8 cm relief).

    python sim/terrain_demo.py                         # both policies -> results/scene_terrain/
    python sim/terrain_demo.py --policy symmetry
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from course_demo import POLICIES, RLPolicy, arm_hold_torque, yaw_of  # noqa: E402
from scenes.scene_terrain import add_scene_terrain  # noqa: E402
from wbc.robots import ROOT, build_model, go2_z1_official as go2_z1  # noqa: E402
from wbc.stance_wbc import Indexer, initial_state  # noqa: E402

DT, DECIM = 0.002, 10


def run(policy, out):
    import imageio
    cfg = go2_z1()
    info = {}
    m = build_model(cfg, torch_mass=0.0, workpiece=False, render_size=(1600, 900), scene_fn=lambda s: info.update(add_scene_terrain(s)))
    for j in cfg.leg_joints:  # RL training joint dynamics
        dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        m.dof_damping[dof], m.dof_frictionloss[dof] = 0.0, 0.01
    d = mujoco.MjData(m)
    ix = Indexer(m, cfg)
    initial_state(m, d, ix)
    x_start, x_goal = info["x0"] + 0.6, info["x1"] - 0.8
    d.qpos[0:2] = [x_start, 0.0]
    d.qpos[2] += 0.09                    # above the highest bumps
    mujoco.mj_forward(m, d)
    rl = RLPolicy(policy, m, d, ix)

    ren = mujoco.Renderer(m, 540, 960)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, ix.base
    cam.distance, cam.azimuth, cam.elevation = 3.0, 70, -15
    frames, log, ev, t_stop = [], dict(cmd=[], v=[], g=[]), {}, None
    for k in range(int(80.0 / DT)):
        x, y = d.qpos[0], d.qpos[1]
        R = d.xmat[ix.base].reshape(3, 3)
        if R[2, 2] < 0.5:
            ev["fell_at"] = round(d.time, 2); ev["fell_x"] = round(float(x), 2)
            break
        if k % DECIM == 0:
            if x < x_goal:
                vx = 0.5 if x < x_goal - 0.4 else max(0.15, 0.5 * (x_goal - x) / 0.4)
                cmd = np.array([vx, 0.0, float(np.clip(-2.0 * yaw_of(d.qpos[3:7]) - 1.5 * y, -0.5, 0.5))])
            else:
                cmd = np.zeros(3)
                if t_stop is None:
                    t_stop = d.time; ev["reached_end_s"] = round(d.time, 2)
            rl.update(cmd)
            if t_stop is None and d.time > 1.0:
                log["cmd"].append(cmd); log["v"].append(R.T @ d.qvel[0:3]); log["g"].append(R.T @ np.array([0, 0, -1.0]))
        tau = np.zeros(m.nv)
        tau[ix.leg_v] = rl.leg_torque()
        tau[ix.arm_v] = arm_hold_torque(d, ix, ix.arm_stow)
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(m, d)
        if k % DECIM == DECIM - 1:
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())
        if t_stop is not None and d.time - t_stop > 2.0:
            break
    c, v, g = (np.array(log[k]) for k in ("cmd", "v", "g"))
    rp = np.degrees(np.arctan2(np.abs(g[:, :2]), -g[:, 2:3])) if len(g) else np.zeros((1, 2))
    ev.update(policy=policy, completed="fell_at" not in ev and t_stop is not None, distance_m=round(float(d.qpos[0] - x_start), 2),
              lin_vel_err=round(float(np.mean(np.linalg.norm(c[:, :2] - v[:, :2], axis=1))), 3) if len(c) else None,
              max_roll_deg=round(float(rp[:, 1].max()), 1), max_pitch_deg=round(float(rp[:, 0].max()), 1),
              terrain_relief_m=round(info["relief_m"], 3))
    imageio.mimsave(os.path.join(out, f"walk__{policy}.mp4"), frames, fps=50, quality=7)
    return ev, frames


def main():
    import imageio
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="moe_cts,symmetry")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "scene_terrain")
    os.makedirs(out, exist_ok=True)
    rows, vids = [], []
    for p in args.policy.split(","):
        ev, frames = run(p, out)
        print(json.dumps(ev)); rows.append(ev); vids.append(frames)
    if len(vids) > 1:
        n = max(len(v) for v in vids)
        vids = [v + [v[-1]] * (n - len(v)) for v in vids]
        imageio.mimsave(os.path.join(out, "walk__comparison.mp4"), [np.concatenate([v[i] for v in vids], 1) for i in range(n)],
                        fps=50, quality=7)
    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
