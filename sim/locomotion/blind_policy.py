"""Blind (proprioceptive) Go2 policies - wty-yy MoE-CTS 82.5k / robot_lab symmetry 77k (policies/go2/, TorchScript) -
behind the same interface as sim/locomotion/policy.py's VelocityPolicy (torques(m, d, cmd) every physics step).

Both were trained on rough terrain + stairs and climb 10 cm steps with the Z1 on board (sim/locomotion_demo.py);
the height-scan "go2_payload" policy was fine-tuned on flat ground and stops at the first step.

    obs (45) = [ang_vel_b * 0.25, projected_gravity_b, cmd * cmd_scale, q - q0, qd * 0.05, last_action]
    joints in Menagerie (depth-first) order, 50 Hz, target = q0 + 0.25 * action, PD kp 25 / kd 0.5
"""
import os

import mujoco
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLIND = {
    "moe_cts": dict(file="moe_cts_82.5k/exported/policy.pt", cmd_scale=[0.5, 1.0, 1.0]),
    "symmetry": dict(file="symmetry_77k/exported/policy.pt", cmd_scale=[1.0, 1.0, 1.0]),
}
KP, KD, ACT_SCALE, TAU_MAX = 25.0, 0.5, 0.25, 23.5


class BlindPolicy:
    def __init__(self, m, cfg, name="symmetry", arm_kp=150.0, arm_kd=8.0, policy_dt=0.02):
        torch.set_num_threads(1)
        self.c = BLIND[name]
        self.name = name
        self.net = torch.jit.load(os.path.join(ROOT, "policies", "go2", self.c["file"]), map_location="cpu").eval()
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.q_idx = np.array([m.jnt_qposadr[J(n)] for n in cfg.leg_joints])
        self.v_idx = np.array([m.jnt_dofadr[J(n)] for n in cfg.leg_joints])
        self.q0 = np.array([cfg.leg_q0[n] for n in cfg.leg_joints])
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
        self.policy_dt = policy_dt
        self.reset()

    @property
    def decimation(self):
        return self._dec

    def reset(self):
        self.last_a = np.zeros(12)
        self.target = self.q0.copy()
        self.primed = False
        self.k = 0
        self._dec = None
        if hasattr(self.net, "reset_memory"):
            self.net.reset_memory()

    def _obs(self, d, cmd):
        R = d.xmat[self.base].reshape(3, 3)
        return np.concatenate([d.qvel[3:6] * 0.25,   # free-joint angular velocity is already in the body frame
                               R.T @ np.array([0, 0, -1.0]), np.asarray(cmd) * np.array(self.c["cmd_scale"]),
                               d.qpos[self.q_idx] - self.q0, d.qvel[self.v_idx] * 0.05, self.last_a])

    def torques(self, m, d, cmd):
        if self._dec is None:
            self._dec = max(1, int(round(self.policy_dt / m.opt.timestep)))
        if self.k % self._dec == 0:
            o = torch.tensor(self._obs(d, cmd), dtype=torch.float32)[None]
            with torch.no_grad():
                if not self.primed:          # fill the policy's internal observation history
                    for _ in range(9):
                        self.net(o)
                    self.primed = True
                a = np.clip(self.net(o)[0].numpy().astype(float), -100, 100)
            self.last_a = a
            self.target = self.q0 + ACT_SCALE * a
        self.k += 1
        tau = KP * (self.target - d.qpos[self.q_idx]) - KD * d.qvel[self.v_idx]
        d.ctrl[self.leg_act] = np.clip(tau, -TAU_MAX, TAU_MAX)
        tau_arm = (self.arm_kp * (self.arm_stow - d.qpos[self.arm_q]) - self.arm_kd * d.qvel[self.arm_v]
                   + d.qfrc_bias[self.arm_v])
        d.ctrl[self.arm_act] = np.clip(tau_arm, -self.arm_tau_max, self.arm_tau_max)
