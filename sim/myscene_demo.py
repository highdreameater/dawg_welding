"""Go2 + Z1 (arm folded) walking in the myscene.blend scene: from the far end of the rock patches towards the ship.

    python sim/myscene_demo.py                    # symmetry policy -> results/myscene/
    python sim/myscene_demo.py --policy moe_cts
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from course_demo import RLPolicy, arm_hold_torque, yaw_of  # noqa: E402
from scenes.myscene import add_myscene  # noqa: E402
from wbc.robots import ROOT, build_model, go2_z1_official as go2_z1  # noqa: E402
from wbc.stance_wbc import Indexer, initial_state  # noqa: E402

DT, DECIM = 0.002, 10
START, LANE_Y, GOAL_X = (30.0, 1.0), 1.0, 16.9   # walk -x over the rocks, climb the 8-step stair, stop on the dock by the ship


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    import imageio
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="symmetry")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "myscene")
    os.makedirs(out, exist_ok=True)

    cfg = go2_z1()
    info = {}
    m = build_model(cfg, torch_mass=0.0, workpiece=False, render_size=(1600, 900), scene_fn=lambda s: info.update(add_myscene(s)))
    for j in cfg.leg_joints:  # RL training joint dynamics
        dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        m.dof_damping[dof], m.dof_frictionloss[dof] = 0.0, 0.01
    d = mujoco.MjData(m)
    ix = Indexer(m, cfg)
    initial_state(m, d, ix)
    d.qpos[0:2] = START
    d.qpos[2] += info["floor_z"]
    d.qpos[3:7] = [0, 0, 0, 1]          # facing -x (towards the ship)
    mujoco.mj_forward(m, d)
    rl = RLPolicy(args.policy, m, d, ix)

    ren = mujoco.Renderer(m, 540, 960)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, ix.base
    cam.distance, cam.azimuth, cam.elevation = 5.0, 160, -12   # behind the robot, ship ahead
    wide = mujoco.MjvCamera()
    wide.lookat[:], wide.distance, wide.azimuth, wide.elevation = [19, 1, 1.5], 22, 140, -20
    ren.update_scene(d, camera=wide)
    imageio.imwrite(os.path.join(out, "overview.png"), ren.render())

    frames, ev, t_stop = [], {}, None
    for k in range(int(60.0 / DT)):
        x, y = d.qpos[0], d.qpos[1]
        if d.xmat[ix.base].reshape(3, 3)[2, 2] < 0.5:
            ev["fell_at"] = round(d.time, 2)
            break
        if k % DECIM == 0:
            yaw_err = wrap(yaw_of(d.qpos[3:7]) - np.pi)
            if x > GOAL_X:
                vx = 0.5 if x > GOAL_X + 0.4 else max(0.15, 0.5 * (x - GOAL_X) / 0.4)
                # facing -x: robot's left is -y, so a +y offset needs a left (positive) turn
                cmd = np.array([vx, 0.0, float(np.clip(-2.0 * yaw_err + 1.5 * (y - LANE_Y), -0.5, 0.5))])
            else:
                cmd = np.zeros(3)
                if t_stop is None:
                    t_stop = d.time; ev["reached_s"] = round(d.time, 2)
            rl.update(cmd)
        tau = np.zeros(m.nv)
        tau[ix.leg_v] = rl.leg_torque()
        tau[ix.arm_v] = arm_hold_torque(d, ix, ix.arm_stow)
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(m, d)
        if k % DECIM == DECIM - 1:
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())
        if t_stop is not None and d.time - t_stop > 3.0:
            break
    ev.update(policy=args.policy, final_pos=np.round(d.qpos[0:3], 2).tolist(), completed="fell_at" not in ev and t_stop is not None)
    imageio.mimsave(os.path.join(out, f"walk__{args.policy}.mp4"), frames, fps=50, quality=7)
    print(json.dumps(ev))


if __name__ == "__main__":
    main()
