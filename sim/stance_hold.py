"""Stance-hold references for the existing stance WBC (sim/wbc/stance_wbc.py): stand still and rigid at a target.

Used by sim/aruco_mission.py when the RL policy hands over at a marker: the WBC levels the base, braces it
`cfg.brace_dz` down, holds it rigid while the head camera re-checks the marker, then raises it back to the walking
height before handing back to the RL policy. The arm stays stowed (joint posture task). No new control law: this
only generates the `ref` dict that StanceWBC.compute() already takes (same convention as force_task.py).

    hold = StanceHold(cfg, qpos0, t0)
    while not hold.done(t): tau, f = wbc.compute(qpos, qvel, hold.ref(t)) ...
"""
import mujoco
import numpy as np

from force_task import min_jerk

PHASES = ["settle", "brace", "hold", "rise"]


class StanceHold:
    def __init__(self, cfg, qpos0, t0, T=dict(settle=0.4, brace=0.8, hold=1.2, rise=0.8)):
        self.t0, self.T = t0, dict(T)
        self.stow = np.asarray(cfg.arm_stow, float)
        self.base_pos0, self.base_quat0 = qpos0[0:3].copy(), qpos0[3:7].copy()
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, self.base_quat0)
        yaw = np.arctan2(R0[3], R0[0])
        self.level_err = np.zeros(3)
        mujoco.mju_subQuat(self.level_err, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), self.base_quat0)
        self.brace_dz = cfg.brace_dz
        t, self.start = 0.0, {}
        for p in PHASES:
            self.start[p] = t
            t += self.T[p]
        self.duration = t

    def phase(self, t):
        tr = t - self.t0
        return [p for p in PHASES if tr >= self.start[p]][-1] if tr >= 0 else PHASES[0]

    def done(self, t):
        return t - self.t0 >= self.duration

    def ref(self, t):
        tr = t - self.t0
        s_down = min_jerk((tr - self.start["brace"]) / self.T["brace"])
        s_up = min_jerk((tr - self.start["rise"]) / self.T["rise"])
        qb = self.base_quat0.copy()
        mujoco.mju_quatIntegrate(qb, self.level_err, s_down)        # level the base (keeps the heading)
        n = len(self.stow)
        return dict(base_pos=self.base_pos0 - np.array([0, 0, self.brace_dz * (s_down - s_up)]), base_quat=qb,
                    arm_q=self.stow, arm_qd=np.zeros(n), arm_qdd=np.zeros(n), w_ee=0.0, w_post=10.0)
