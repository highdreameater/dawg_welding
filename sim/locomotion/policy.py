"""Isaac Lab velocity-tracking locomotion policies (RSL-RL actors) running in MuJoCo, with the arm stowed as payload.

Supported (LOCO):
    anymal_c  policies/anymal/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt   ANYdrive LSTM actuator net, scale 0.5
    go2       archive/other_robots/policies/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt
              DC motor PD (kp 25, kd 0.5, 23.5 Nm, 30 rad/s), action scale 0.25 (Isaac Lab 2.3 UNITREE_GO2_CFG)
    go2_payload  a Go2 policy fine-tuned in MuJoCo with the Z1 on board (sim/locomotion/train_payload.py)

Policy interface (Isaac Lab LocomotionVelocityRoughEnvCfg):
    50 Hz policy (sim dt 0.005 s, decimation 4), 235-D observation, 12-D action, no observation normalisation
    obs = [ base_lin_vel_b (3), base_ang_vel_b (3), projected_gravity_b (3), cmd (vx, vy, wz) (3),
            q - q0 (12), qd (12), last_action (12), height_scan (187) ]          joints in Isaac (breadth-first) order
    height_scan = base_z - hit_z - 0.5, clipped to [-1, 1]; 17 x 11 grid, 0.1 m, 1.6 m x 1.0 m, yaw-aligned,
                  x varies fastest. (Measured from the base, *not* the +20 m sensor offset: with the offset the
                  pretrained ANYmal policy does not walk at all.)
    target_q = q0 + scale * action -> actuator model -> torque, DC-motor torque/speed clipped
"""
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY_DT, SIM_DT = 0.02, 0.005
SCAN_RES, SCAN_SIZE = 0.1, (1.6, 1.0)


@dataclass
class LocoCfg:
    checkpoint: str
    joints: list               # policy (Isaac) joint order
    action_scale: float
    actuator: str              # "lstm" (actuator net) or "dc" (PD + DC-motor clip)
    sat_effort: object         # scalar or per-joint array (policy order)
    effort_limit: object
    vel_limit: object
    kp: float = 0.0
    kd: float = 0.0
    actuator_net: str = None


def _breadth_first(legs, parts):
    return [f"{l}_{p}" for p in parts for l in legs]


LOCO = {
    "anymal_c": LocoCfg(
        checkpoint=os.path.join(ROOT, "policies", "anymal", "Isaac-Velocity-Rough-Anymal-C-v0", "checkpoint.pt"),
        joints=_breadth_first(("LF", "LH", "RF", "RH"), ("HAA", "HFE", "KFE")),
        action_scale=0.5, actuator="lstm", sat_effort=120.0, effort_limit=80.0, vel_limit=7.5,
        actuator_net=os.path.join(ROOT, "locomotion", "legged_gym", "resources", "actuator_nets", "anydrive_v3_lstm.pt")),
    "go2": LocoCfg(
        checkpoint=os.path.join(ROOT, "archive", "other_robots", "policies", "Isaac-Velocity-Rough-Unitree-Go2-v0",
                                "checkpoint.pt"),
        joints=_breadth_first(("FL", "FR", "RL", "RR"), ("hip_joint", "thigh_joint", "calf_joint")),
        action_scale=0.25, actuator="dc", sat_effort=23.5, effort_limit=23.5, vel_limit=30.0, kp=25.0, kd=0.5),
}
LOCO["go2_payload"] = LocoCfg(**{**LOCO["go2"].__dict__,
                                 "checkpoint": os.path.join(ROOT, "policies", "go2", "go2_z1_payload", "checkpoint.pt")})
ROBOT_LOCO = {"anymal_c_z1": "anymal_c", "go2_z1": "go2"}


def build_actor(state_dict):
    keys = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for i, idx in enumerate(keys):
        w = state_dict[f"actor.{idx}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(state_dict[f"actor.{idx}.bias"])
        layers += [lin] + ([nn.ELU()] if i < len(keys) - 1 else [])
    return nn.Sequential(*layers).eval()


def scan_grid():
    x = np.arange(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2 + 1e-9, SCAN_RES)
    y = np.arange(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2 + 1e-9, SCAN_RES)
    gx, gy = np.meshgrid(x, y, indexing="xy")  # (len(y), len(x)) -> x fastest after flatten
    return np.stack([gx.ravel(), gy.ravel()], 1)  # (187, 2)


def dc_clip(tau, qd, sat, lim, vlim):
    """Isaac Lab DCMotor torque-speed envelope."""
    hi = np.clip(sat * (1.0 - qd / vlim), 0.0, lim)
    lo = np.clip(sat * (-1.0 - qd / vlim), -lim, 0.0)
    return np.clip(tau, lo, hi)


class VelocityPolicy:
    """Call `torques(m, d, cmd)` every physics step (dt = SIM_DT). It re-plans the action at 50 Hz and writes d.ctrl
    for the legs (policy + actuator model) and the arm (stowed: joint PD + gravity compensation)."""

    def __init__(self, m, cfg, loco="auto", checkpoint=None, arm_kp=150.0, arm_kd=8.0, joints=None):
        torch.set_num_threads(1)
        self.lc = LOCO[ROBOT_LOCO[cfg.name] if loco == "auto" else loco]
        ck = torch.load(checkpoint or self.lc.checkpoint, map_location="cpu", weights_only=False)
        self.actor = build_actor(ck["model_state_dict"])
        self.net = torch.jit.load(self.lc.actuator_net, map_location="cpu").eval() if self.lc.actuator == "lstm" else None
        self.joints = joints or self.lc.joints
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.q_idx = np.array([m.jnt_qposadr[J(n)] for n in self.joints])
        self.v_idx = np.array([m.jnt_dofadr[J(n)] for n in self.joints])
        self.q0 = np.array([cfg.leg_q0[n] for n in self.joints])
        arm = cfg.arm_names()
        self.arm_q = np.array([m.jnt_qposadr[J(n)] for n in arm])
        self.arm_v = np.array([m.jnt_dofadr[J(n)] for n in arm])
        self.arm_stow = np.asarray(cfg.arm_stow, float)
        self.arm_tau_max = np.array([cfg.tau_limits[n] for n in arm])
        self.arm_kp, self.arm_kd = arm_kp, arm_kd
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body)
        act_of_dof = {m.jnt_dofadr[m.actuator_trnid[a, 0]]: a for a in range(m.nu)}
        self.leg_act = np.array([act_of_dof[v] for v in self.v_idx])
        self.arm_act = np.array([act_of_dof[v] for v in self.arm_v])
        self.grid = scan_grid()
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # terrain (floor) only: obstacles are group 1
        self.decimation = int(round(POLICY_DT / SIM_DT))
        self.reset()

    def reset(self):
        self.h = torch.zeros(2, 12, 8)
        self.c = torch.zeros(2, 12, 8)
        self.action = np.zeros(12)
        self.target = self.q0.copy()
        self.k = 0

    # ------------------------------------------------------------------ observation
    def height_scan(self, m, d):
        p = d.xpos[self.base]
        R = d.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        pts = p[:2] + self.grid @ np.array([[c, s], [-s, c]])
        out = np.empty(len(pts))
        gid = np.zeros(1, np.int32)
        down = np.array([0.0, 0.0, -1.0])
        for i, (x, y) in enumerate(pts):
            dist = mujoco.mj_ray(m, d, np.array([x, y, p[2] + 20.0]), down, self.geomgroup, 1, -1, gid)
            hit_z = p[2] + 20.0 - dist if dist >= 0 else -1e3
            out[i] = p[2] - hit_z - 0.5
        return np.clip(out, -1.0, 1.0)

    def observe(self, m, d, cmd):
        R = d.xmat[self.base].reshape(3, 3)
        return np.concatenate([R.T @ d.qvel[0:3], d.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]), cmd,
                               d.qpos[self.q_idx] - self.q0, d.qvel[self.v_idx], self.action, self.height_scan(m, d)])

    # ------------------------------------------------------------------ control
    def torques(self, m, d, cmd):
        lc = self.lc
        if self.k % self.decimation == 0:
            obs = torch.as_tensor(self.observe(m, d, np.asarray(cmd, float)), dtype=torch.float32)
            with torch.inference_mode():
                self.action = self.actor(obs).numpy().astype(float)
            self.target = self.q0 + lc.action_scale * self.action
        self.k += 1
        q, qd = d.qpos[self.q_idx], d.qvel[self.v_idx]
        if lc.actuator == "lstm":
            x = torch.tensor(np.stack([self.target - q, qd], 1)[:, None, :], dtype=torch.float32)  # (12, 1, 2)
            with torch.inference_mode():
                tau, (self.h, self.c) = self.net(x, (self.h, self.c))
            tau = tau.numpy().astype(float).reshape(12)
        else:
            tau = lc.kp * (self.target - q) - lc.kd * qd
        d.ctrl[self.leg_act] = dc_clip(tau, qd, lc.sat_effort, lc.effort_limit, lc.vel_limit)
        tau_arm = (self.arm_kp * (self.arm_stow - d.qpos[self.arm_q]) - self.arm_kd * d.qvel[self.arm_v]
                   + d.qfrc_bias[self.arm_v])
        d.ctrl[self.arm_act] = np.clip(tau_arm, -self.arm_tau_max, self.arm_tau_max)
