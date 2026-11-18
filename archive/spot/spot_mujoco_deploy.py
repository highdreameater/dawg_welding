"""Deploy pretrained Isaac Lab / Isaac Sim Spot locomotion policies in MuJoCo (sim-to-sim), zero-shot.

Robot models : MuJoCo Menagerie boston_dynamics_spot (spot.xml / spot_arm.xml)
Policies     : 48-D obs, 12-D action, 50 Hz (dt=0.002, decimation=10) - Isaac-Velocity-Flat-Spot-v0 spec
    obs = [base_lin_vel_b(3), base_ang_vel_b(3), projected_gravity_b(3), cmd(vx,vy,wz)(3),
           joint_pos - default(12), joint_vel(12), last_action(12)]     (joints in Isaac/"orbit" order)
    target_q = default_q + 0.2 * action   ->  PD: kp=60, kd=1.5, effort hip 45 Nm / knee 115 Nm

Examples:
    python archive/spot/spot_mujoco_deploy.py --robot spot_arm --policy isaacsim --scenario walk
    python archive/spot/spot_mujoco_deploy.py --all                 # full matrix -> results/summary.md
    mjpython archive/spot/spot_mujoco_deploy.py --robot spot_arm --scenario arm_sweep --viewer   # live window (macOS needs mjpython)
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(ROOT, "third_party", "mujoco_menagerie", "boston_dynamics_spot")
POLICIES = {
    "isaacsim": os.path.join(HERE, "policies", "isaacsim_spot_flat", "spot_policy.pt"),
    "isaaclab": os.path.join(HERE, "policies", "isaaclab_flat_spot_rslrl", "policy.pt"),
}
ROBOTS = {"spot": "scene.xml", "spot_arm": "scene_arm.xml"}

# Isaac Lab (orbit) joint order used by the policy
ORBIT_JOINTS = ["fl_hx", "fr_hx", "hl_hx", "hr_hx", "fl_hy", "fr_hy", "hl_hy", "hr_hy", "fl_kn", "fr_kn", "hl_kn", "hr_kn"]
DEFAULT_Q = {"fl_hx": 0.1, "hl_hx": 0.1, "fr_hx": -0.1, "hr_hx": -0.1,
             "fl_hy": 0.9, "fr_hy": 0.9, "hl_hy": 1.1, "hr_hy": 1.1,
             "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5}
KP, KD, ACTION_SCALE, DECIMATION = 60.0, 1.5, 0.2, 10
EFFORT = {"hx": 45.0, "hy": 45.0, "kn": 115.0}

ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]
ARM_STOW = np.array([0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0])
ARM_REACH = np.array([0.0, -0.35, 0.55, 0.0, -0.2, 0.0, 0.0])  # arm out in front of the body (welding-like pose)

SCENARIOS = ["stand", "walk", "arm_reach", "arm_sweep", "rough", "stairs"]


# ----------------------------------------------------------------------------- scenario definitions
def command(scn, t):
    """(vx, vy, wz) velocity command for the base."""
    if scn == "stand":
        return np.zeros(3)
    if scn in ("stairs", "rough"):
        return np.array([0.6, 0.0, 0.0]) if t > 1.0 else np.zeros(3)
    # walk / arm_reach / arm_sweep: same command profile so results are comparable
    sched = [(1.5, (0, 0, 0)), (5.5, (1.0, 0, 0)), (8.5, (0, 0, 1.0)), (11.5, (0, 0.5, 0)),
             (14.5, (-0.6, 0, 0)), (1e9, (0, 0, 0))]
    for t_end, c in sched:
        if t < t_end:
            return np.array(c, dtype=float)


def arm_target(scn, t):
    if scn in ("arm_reach", "arm_sweep"):
        a = np.clip((t - 0.5) / 1.5, 0, 1)  # unstow over 1.5 s
        q = (1 - a) * ARM_STOW + a * ARM_REACH
        if scn == "arm_sweep" and t > 2.0:  # welding-like sweeping motion: yaw + pitch oscillations
            q = q.copy()
            q[0] += 0.9 * np.sin(2 * np.pi * 0.3 * (t - 2.0))
            q[1] += 0.3 * np.sin(2 * np.pi * 0.5 * (t - 2.0))
        return q
    return ARM_STOW


def duration(scn):
    return {"stand": 6.0, "walk": 16.0, "arm_reach": 16.0, "arm_sweep": 16.0, "rough": 12.0, "stairs": 12.0}[scn]


# ----------------------------------------------------------------------------- model construction
def build_model(robot, scn, step_height, seed):
    spec = mujoco.MjSpec.from_file(os.path.join(MENAGERIE, ROBOTS[robot]))
    wb = spec.worldbody
    if scn == "stairs":
        run, n = 0.30, 8
        x0 = 1.5
        for i in range(n):
            h = step_height * (i + 1)
            g = wb.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [run / 2, 1.5, h / 2]
            g.pos = [x0 + run * i + run / 2, 0, h / 2]
            g.rgba = [0.55, 0.55, 0.6, 1]
        top = wb.add_geom()
        top.type = mujoco.mjtGeom.mjGEOM_BOX
        top.size = [2.0, 1.5, step_height * n / 2]
        top.pos = [x0 + run * n + 2.0, 0, step_height * n / 2]
        top.rgba = [0.55, 0.55, 0.6, 1]
    elif scn == "rough":
        rng = np.random.default_rng(seed)
        s = 0.25
        for ix in range(28):
            for iy in range(12):
                h = rng.uniform(0.0, 0.06)
                g = wb.add_geom()
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size = [s / 2, s / 2, h / 2 + 1e-3]
                g.pos = [0.8 + ix * s, (iy - 6) * s + s / 2, h / 2]
                c = 0.4 + h * 4
                g.rgba = [c, c, c + 0.05, 1]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    model = spec.compile()
    model.opt.timestep = 0.002

    # Make leg actuators match the training actuator model (PD kp=60 kd=1.5 with effort limits)
    for j in ORBIT_JOINTS:
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        model.actuator_gainprm[a, 0] = KP
        model.actuator_biasprm[a, 1] = -KP
        model.actuator_biasprm[a, 2] = -KD
        lim = EFFORT[j.split("_")[1]]
        model.actuator_forcelimited[a] = 1
        model.actuator_forcerange[a] = [-lim, lim]
    return model


# ----------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, robot, policy, scn, step_height=0.10, seed=0):
        self.robot, self.policy_name, self.scn = robot, policy, scn
        self.model = build_model(robot, scn, step_height, seed)
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(POLICIES[policy], map_location="cpu").eval()
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ORBIT_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ORBIT_JOINTS])
        self.leg_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ORBIT_JOINTS])
        self.has_arm = robot == "spot_arm"
        if self.has_arm:
            self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS])
            self.arm_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body")
        self.default_q = np.array([DEFAULT_Q[j] for j in ORBIT_JOINTS])
        self.last_action = np.zeros(12)
        self.reset()

    def reset(self):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[0:3] = [0, 0, 0.55]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = self.default_q
        d.ctrl[self.leg_act] = self.default_q
        if self.has_arm:
            d.qpos[self.arm_qadr] = ARM_STOW
            d.ctrl[self.arm_act] = ARM_STOW
        mujoco.mj_forward(self.model, d)
        self.last_action[:] = 0

    def obs(self, cmd):
        d = self.data
        R = d.xmat[self.base].reshape(3, 3)
        v_b = R.T @ d.qvel[0:3]
        w_b = d.qvel[3:6].copy()  # free-joint angular velocity is already in the body frame
        g_b = R.T @ np.array([0, 0, -1.0])
        o = np.concatenate([v_b, w_b, g_b, cmd, d.qpos[self.qadr] - self.default_q, d.qvel[self.dadr], self.last_action])
        return o.astype(np.float32), v_b, w_b, g_b

    def base_touching_ground(self):
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == self.base and b2 == 0) or (b2 == self.base and b1 == 0):
                return True
        return False

    def policy_step(self):
        t = self.data.time
        cmd = command(self.scn, t)
        o, v_b, w_b, g_b = self.obs(cmd)
        with torch.no_grad():
            a = self.policy(torch.from_numpy(o)[None])[0].numpy().astype(np.float64)
        self.last_action = a
        self.data.ctrl[self.leg_act] = self.default_q + ACTION_SCALE * a
        if self.has_arm:
            self.data.ctrl[self.arm_act] = arm_target(self.scn, t)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        return cmd, v_b, w_b, g_b

    def total_com(self):
        return self.data.subtree_com[self.base].copy()


def run_episode(robot, policy, scn, video=None, viewer=False, step_height=0.10, realtime=False):
    r = Runner(robot, policy, scn, step_height)
    T = duration(scn)
    n_steps = int(T / (DECIMATION * r.model.opt.timestep))
    renderer = cam = frames = None
    if video:
        renderer = mujoco.Renderer(r.model, 480, 720)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = r.base
        cam.distance, cam.azimuth, cam.elevation = 3.0, 140, -18
        frames = []
    v = None
    if viewer:
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch_passive(r.model, r.data)
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = r.base
        v.cam.distance = 3.0

    log = {"t": [], "cmd": [], "v_b": [], "w_b": [], "g_b": [], "z": [], "com_offset_xy": []}
    fell_at = None
    for k in range(n_steps):
        t0 = time.time()
        cmd, v_b, w_b, g_b = r.policy_step()
        com = r.total_com()
        base_pos = r.data.xpos[r.base]
        R = r.data.xmat[r.base].reshape(3, 3)
        log["t"].append(r.data.time); log["cmd"].append(cmd); log["v_b"].append(v_b); log["w_b"].append(w_b)
        log["g_b"].append(g_b); log["z"].append(base_pos[2])
        log["com_offset_xy"].append((R.T @ (com - base_pos))[:2])
        if fell_at is None and (g_b[2] > -0.5 or r.base_touching_ground()):
            fell_at = r.data.time
        if renderer is not None:
            renderer.update_scene(r.data, camera=cam)
            frames.append(renderer.render())
        if v is not None:
            if not v.is_running():
                break
            v.sync()
        if (v is not None or realtime):
            time.sleep(max(0, DECIMATION * r.model.opt.timestep - (time.time() - t0)))
        if fell_at is not None and r.data.time - fell_at > 2.0 and v is None:
            break  # keep 2 s after the fall for the video, then stop

    if v is not None:
        v.close()
    if renderer is not None and frames:
        import imageio
        os.makedirs(os.path.dirname(video), exist_ok=True)
        imageio.mimsave(video, frames, fps=int(1 / (DECIMATION * r.model.opt.timestep)), quality=7)
    return summarize(robot, policy, scn, log, fell_at, T, video)


def summarize(robot, policy, scn, log, fell_at, T, video):
    t = np.array(log["t"]); cmd = np.array(log["cmd"]); v = np.array(log["v_b"]); w = np.array(log["w_b"])
    g = np.array(log["g_b"]); com = np.array(log["com_offset_xy"])
    valid = t < (fell_at if fell_at is not None else 1e9)
    moving = valid & (np.linalg.norm(cmd, axis=1) > 0)
    roll = np.degrees(np.arctan2(g[:, 1], -g[:, 2]))
    pitch = np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
    res = {
        "robot": robot, "policy": policy, "scenario": scn,
        "survived_s": float(fell_at if fell_at is not None else t[-1]), "duration_s": T, "fell": fell_at is not None,
        "lin_vel_err_mps": float(np.mean(np.linalg.norm(cmd[moving, :2] - v[moving, :2], axis=1))) if moving.any() else None,
        "yaw_rate_err_rps": float(np.mean(np.abs(cmd[moving, 2] - w[moving, 2]))) if moving.any() else None,
        "max_abs_roll_deg": float(np.max(np.abs(roll[valid]))) if valid.any() else None,
        "max_abs_pitch_deg": float(np.max(np.abs(pitch[valid]))) if valid.any() else None,
        "mean_base_z_m": float(np.mean(np.array(log["z"])[valid])) if valid.any() else None,
        "max_com_offset_xy_m": float(np.max(np.linalg.norm(com, axis=1))),
        "video": os.path.relpath(video, ROOT) if video else None,
    }
    return res


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="spot_arm")
    ap.add_argument("--policy", choices=list(POLICIES), default="isaacsim")
    ap.add_argument("--scenario", choices=SCENARIOS, default="walk")
    ap.add_argument("--step-height", type=float, default=0.10)
    ap.add_argument("--viewer", action="store_true", help="interactive window (use `mjpython` on macOS)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--all", action="store_true", help="run every robot x policy x scenario, write results/summary.md")
    args = ap.parse_args()
    out = os.path.join(HERE, "results", "pretrained_locomotion")

    if args.all:
        rows = []
        for policy in POLICIES:
            for robot in ROBOTS:
                for scn in SCENARIOS:
                    if robot == "spot" and scn.startswith("arm"):
                        continue
                    vid = None if args.no_video else os.path.join(out, "videos", f"{robot}__{policy}__{scn}.mp4")
                    res = run_episode(robot, policy, scn, video=vid, step_height=args.step_height)
                    print(json.dumps(res))
                    rows.append(res)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(rows, f, indent=2)
        hdr = "| policy | robot | scenario | survived (s) | fell | lin-vel err (m/s) | yaw-rate err (rad/s) | max roll (deg) | max pitch (deg) | base z (m) | max CoM xy offset (m) | video |\n|" + "---|" * 12
        lines = [hdr] + [
            f"| {r['policy']} | {r['robot']} | {r['scenario']} | {fmt(r['survived_s'],1)}/{fmt(r['duration_s'],0)} | {'**YES**' if r['fell'] else 'no'} | "
            f"{fmt(r['lin_vel_err_mps'])} | {fmt(r['yaw_rate_err_rps'])} | {fmt(r['max_abs_roll_deg'],1)} | {fmt(r['max_abs_pitch_deg'],1)} | "
            f"{fmt(r['mean_base_z_m'])} | {fmt(r['max_com_offset_xy_m'],3)} | {r['video'] or '-'} |" for r in rows]
        with open(os.path.join(out, "summary.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    vid = None if (args.no_video or args.viewer) else os.path.join(out, "videos", f"{args.robot}__{args.policy}__{args.scenario}.mp4")
    res = run_episode(args.robot, args.policy, args.scenario, video=vid, viewer=args.viewer, step_height=args.step_height)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
