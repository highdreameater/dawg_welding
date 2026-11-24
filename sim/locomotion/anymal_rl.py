"""Pretrained Isaac Lab ANYmal-C rough-terrain policy (Isaac-Velocity-Rough-Anymal-C-v0, RSL-RL) running in MuJoCo.

Sim-to-sim deployment of policies/anymal/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt on the ANYmal-C + Z1 model
(arm stowed, held by a joint PD -> a static ~5.6 kg payload on the front of the base).

Policy interface (Isaac Lab LocomotionVelocityRoughEnvCfg + AnymalCRoughEnvCfg):
    50 Hz policy (sim dt 0.005 s, decimation 4), 235-D observation, 12-D action, no observation normalisation
    obs = [ base_lin_vel_b (3), base_ang_vel_b (3), projected_gravity_b (3), cmd (vx, vy, wz) (3),
            q - q0 (12), qd (12), last_action (12), height_scan (187) ]          joints in Isaac (breadth-first) order
    height_scan = sensor_z - hit_z - 0.5, clipped to [-1, 1]; 17 x 11 grid, 0.1 m, 1.6 m x 1.0 m, yaw-aligned,
                  x varies fastest
    target_q = q0 + 0.5 * action  ->  ANYdrive LSTM actuator net (pos error, vel) -> torque, DC-motor clipped
              (saturation 120 Nm, effort limit 80 Nm, velocity limit 7.5 rad/s)
"""
import os

import mujoco
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "policies", "anymal", "Isaac-Velocity-Rough-Anymal-C-v0", "checkpoint.pt")
ACTUATOR_NET = os.path.join(ROOT, "locomotion", "legged_gym", "resources", "actuator_nets", "anydrive_v3_lstm.pt")

ISAAC_JOINTS = [f"{leg}_{j}" for j in ("HAA", "HFE", "KFE") for leg in ("LF", "LH", "RF", "RH")]
POLICY_DT, SIM_DT, ACTION_SCALE = 0.02, 0.005, 0.5
SAT_EFFORT, EFFORT_LIMIT, VEL_LIMIT = 120.0, 80.0, 7.5
SCAN_RES, SCAN_SIZE = 0.1, (1.6, 1.0)


def _actor(state_dict):
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


class AnymalRoughPolicy:
    """Call `torques(m, d, cmd)` every physics step (dt = 0.005 s). It re-plans the action at 50 Hz."""

    def __init__(self, m, cfg, checkpoint=CHECKPOINT, actuator_net=ACTUATOR_NET, arm_kp=150.0, arm_kd=8.0):
        torch.set_num_threads(1)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.actor = _actor(ck["model_state_dict"])
        self.net = torch.jit.load(actuator_net, map_location="cpu").eval()
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.q_idx = np.array([m.jnt_qposadr[J(n)] for n in ISAAC_JOINTS])
        self.v_idx = np.array([m.jnt_dofadr[J(n)] for n in ISAAC_JOINTS])
        self.q0 = np.array([cfg.leg_q0[n] for n in ISAAC_JOINTS])
        arm = cfg.arm_names()
        self.arm_q = np.array([m.jnt_qposadr[J(n)] for n in arm])
        self.arm_v = np.array([m.jnt_dofadr[J(n)] for n in arm])
        self.arm_stow = np.asarray(cfg.arm_stow, float)
        self.arm_tau_max = np.array([cfg.tau_limits[n] for n in arm])
        self.arm_kp, self.arm_kd = arm_kp, arm_kd
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body)
        # actuator index of every dof (to write d.ctrl)
        self.act_of_dof = {m.jnt_dofadr[m.actuator_trnid[a, 0]]: a for a in range(m.nu)}
        self.leg_act = np.array([self.act_of_dof[v] for v in self.v_idx])
        self.arm_act = np.array([self.act_of_dof[v] for v in self.arm_v])
        self.grid = scan_grid()
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # world geoms only (robot is in groups 2/3)
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
        v_b = R.T @ d.qvel[0:3]
        w_b = d.qvel[3:6].copy()  # free-joint angular velocity is already in the body frame
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([v_b, w_b, g_b, cmd, d.qpos[self.q_idx] - self.q0, d.qvel[self.v_idx],
                               self.action, self.height_scan(m, d)])

    # ------------------------------------------------------------------ control
    def torques(self, m, d, cmd):
        if self.k % self.decimation == 0:
            obs = torch.as_tensor(self.observe(m, d, np.asarray(cmd, float)), dtype=torch.float32)
            with torch.inference_mode():
                self.action = self.actor(obs).numpy().astype(float)
            self.target = self.q0 + ACTION_SCALE * self.action
        self.k += 1
        q, qd = d.qpos[self.q_idx], d.qvel[self.v_idx]
        x = torch.tensor(np.stack([self.target - q, qd], 1)[:, None, :], dtype=torch.float32)  # (12, 1, 2)
        with torch.inference_mode():
            tau, (self.h, self.c) = self.net(x, (self.h, self.c))
        tau = tau.numpy().astype(float).reshape(12)
        # DC-motor torque-speed envelope (Isaac Lab DCMotor / ActuatorNetLSTM._clip_effort)
        hi = np.clip(SAT_EFFORT * (1.0 - qd / VEL_LIMIT), 0.0, EFFORT_LIMIT)
        lo = np.clip(SAT_EFFORT * (-1.0 - qd / VEL_LIMIT), -EFFORT_LIMIT, 0.0)
        d.ctrl[self.leg_act] = np.clip(tau, lo, hi)
        # arm: stowed, joint PD + gravity/Coriolis compensation
        tau_arm = (self.arm_kp * (self.arm_stow - d.qpos[self.arm_q]) - self.arm_kd * d.qvel[self.arm_v]
                   + d.qfrc_bias[self.arm_v])
        d.ctrl[self.arm_act] = np.clip(tau_arm, -self.arm_tau_max, self.arm_tau_max)
