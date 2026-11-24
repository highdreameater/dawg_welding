"""Simulated depth camera (ray-cast) mounted on the robot.

Defaults follow an Intel RealSense D435-class sensor: 87 x 58 deg field of view, 0.15-3 m range, depth noise that
grows with the square of the distance (sigma = 0.5 mm + 2 mm * z^2), here at 160 x 120 rays. Every ray is cast with
`mj_multiRay` from the camera centre against the floor and world geoms (groups 0 and 1); the robot's own geoms
(groups 2 and 3) are invisible to it.
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CameraMount:
    body: str = "base"
    pos: tuple = (0.30, 0.0, 0.05)     # Go2 head, in the base frame
    pitch_deg: float = 25.0            # tilted down


class DepthCamera:
    def __init__(self, m, mount=CameraMount(), width=160, height=120, hfov_deg=87.0, vfov_deg=58.0,
                 z_min=0.15, z_max=3.0, noise_a=0.0005, noise_b=0.002, seed=0):
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mount.body)
        self.pos = np.asarray(mount.pos, float)
        p = np.radians(mount.pitch_deg)
        self.R_mount = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])  # pitch down
        u = np.tan(np.radians(hfov_deg) / 2) * np.linspace(1, -1, width)    # +y (left) ... -y
        v = np.tan(np.radians(vfov_deg) / 2) * np.linspace(1, -1, height)   # +z (up) ... -z
        U, V = np.meshgrid(u, v)
        rays = np.stack([np.ones_like(U), U, V], -1).reshape(-1, 3)
        self.rays_cam = rays / np.linalg.norm(rays, axis=1, keepdims=True)  # camera frame: x forward, y left, z up
        self.shape = (height, width)
        self.z_min, self.z_max = z_min, z_max
        self.noise_a, self.noise_b = noise_a, noise_b
        self.rng = np.random.default_rng(seed)
        self.geomgroup = np.array([1, 1, 0, 0, 0, 0], np.uint8)

    def pose(self, d):
        R_b = d.xmat[self.body].reshape(3, 3)
        return d.xpos[self.body] + R_b @ self.pos, R_b @ self.R_mount

    def capture(self, m, d):
        """Returns (points_world (N, 3), camera_position (3,), depth image (H, W) with NaN where nothing was hit)."""
        c, R = self.pose(d)
        vec = (self.rays_cam @ R.T).ravel().copy()
        n = len(self.rays_cam)
        dist = np.zeros(n)
        gid = np.zeros(n, np.int32)
        mujoco.mj_multiRay(m, d, c, vec, self.geomgroup, 1, -1, gid, dist, None, n, self.z_max)
        ok = (dist > 0) & (gid >= 0)
        # depth along the optical axis; noise grows with depth^2 (stereo depth camera)
        depth = dist * self.rays_cam[:, 0]
        ok &= (depth > self.z_min) & (depth < self.z_max)
        sigma = self.noise_a + self.noise_b * depth ** 2
        dist_noisy = dist + self.rng.normal(0, 1, n) * sigma / np.maximum(self.rays_cam[:, 0], 1e-6)
        pts = c + vec.reshape(-1, 3) * dist_noisy[:, None]
        img = np.where(ok, depth, np.nan).reshape(self.shape)
        return pts[ok], c, img


def scan(m, d, cam, frames=5):
    """Several frames from a standing robot (independent noise), stacked. Returns (points, camera position)."""
    clouds, c = [], None
    for _ in range(frames):
        pts, c, _ = cam.capture(m, d)
        clouds.append(pts)
    return np.concatenate(clouds), c
