"""Path follower: planned 2-D path + goal heading -> body-frame velocity command (vx, vy, wz) for the RL policy.

Pure pursuit on a lookahead point while far from the goal (walk forward, heading along the path), then an
omnidirectional docking controller (position + heading PD) that brings the base to the station pose.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PathFollower:
    def __init__(self, path, goal_yaw, v_max=0.6, vy_max=0.35, wz_max=0.8, lookahead=0.5,
                 dock_radius=0.8, pos_tol=0.03, yaw_tol=np.radians(2.0)):
        self.path, self.goal_yaw = np.asarray(path, float), goal_yaw
        self.goal = self.path[-1]
        self.v_max, self.vy_max, self.wz_max = v_max, vy_max, wz_max
        self.lookahead, self.dock_radius = lookahead, dock_radius
        self.pos_tol, self.yaw_tol = pos_tol, yaw_tol
        self.i = 0          # progress index along the path (monotonic)
        self.done = False

    def command(self, xy, yaw):
        p = np.asarray(xy, float)
        # progress: closest path point within a forward window
        win = self.path[self.i:self.i + 40]
        self.i += int(np.argmin(np.linalg.norm(win - p, axis=1)))
        d_goal = np.linalg.norm(self.goal - p)
        e_yaw_goal = wrap(self.goal_yaw - yaw)
        if d_goal < self.pos_tol and abs(e_yaw_goal) < self.yaw_tol:
            self.done = True
        if self.done:
            return np.zeros(3)

        if d_goal > self.dock_radius:
            # pure pursuit: aim at the lookahead point, face along the path
            s = np.cumsum(np.r_[0, np.linalg.norm(np.diff(self.path[self.i:], axis=0), axis=1)])
            j = self.i + int(np.searchsorted(s, self.lookahead))
            target = self.path[min(j, len(self.path) - 1)]
            e = target - p
            heading = np.arctan2(e[1], e[0])
            e_yaw = wrap(heading - yaw)
            speed = self.v_max * np.clip(1.0 - abs(e_yaw) / np.radians(60), 0.15, 1.0)  # slow down to turn
            v_w = speed * e / (np.linalg.norm(e) + 1e-9)
            wz = 1.5 * e_yaw
        else:
            # docking: omnidirectional PD to the goal pose, heading blends to the goal yaw
            v_w = 1.2 * (self.goal - p)
            nv = np.linalg.norm(v_w)
            v_w *= min(1.0, 0.35 / (nv + 1e-9))
            wz = 1.5 * e_yaw_goal
        c, s = np.cos(yaw), np.sin(yaw)
        v_b = np.array([c * v_w[0] + s * v_w[1], -s * v_w[0] + c * v_w[1]])
        return np.array([np.clip(v_b[0], -self.v_max, self.v_max), np.clip(v_b[1], -self.vy_max, self.vy_max),
                         np.clip(wz, -self.wz_max, self.wz_max)])
