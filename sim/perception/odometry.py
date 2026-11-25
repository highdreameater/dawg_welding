"""Robot pose estimate: drifting leg/IMU odometry, re-anchored on ArUco markers (landmarks).

The physics state is the truth. The robot does not get it: it integrates its body velocity with the errors a real
leg-inertial odometry has, so its belief walks away from the truth while it walks:
    v_meas  = diag(1 + sx, 1 + sy) v_body + white noise            (scale errors: slip, leg kinematics)
    wz_meas = (1 + sw) wz + k_yaw |v| + white noise                  (heading drift grows with distance walked)
Standing still (all feet planted, base still) adds nothing (zero-velocity update).
Defaults: 3 % scale error, 1 % yaw-rate scale error, 0.11 deg/m heading drift (typical leg-inertial odometry).

Everything the robot computes (marker poses, depth points, path following) lives in this *estimated* frame:
    to_est(P) / to_true(P): the rigid 2-D transform (x, y, yaw) between the physics frame and the estimate.

Landmarks: each work-order marker is anchored in the estimated frame (weighted by 1/distance^2, refined only by
views closer than the closest so far - a closer view is more accurate). Every other sighting re-localises the robot:
the pose correction that puts the marker back onto its anchor is applied with a gain (several sightings average
the PnP noise). See MarkerMap in sim/perception/aruco_detect.py.
"""
import numpy as np


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Odometry:
    def __init__(self, xy, yaw, scale=(0.03, -0.03), yaw_scale=0.01, yaw_per_m=0.002, noise_v=0.01, noise_w=0.004,
                 seed=1):
        self.p, self.yaw = np.asarray(xy, float).copy(), float(yaw)
        self.sx, self.sy = scale
        self.sw, self.k_yaw = yaw_scale, yaw_per_m
        self.noise_v, self.noise_w = noise_v, noise_w
        self.rng = np.random.default_rng(seed)
        self.true_p, self.true_yaw = self.p.copy(), self.yaw
        self.corrections = []

    def update(self, d, base_xmat, dt, still=False):
        R = base_xmat.reshape(3, 3)
        self.true_p = d.qpos[0:2].copy()
        self.true_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if still:
            return
        vb = R.T @ d.qvel[0:3]                    # free joint: linear velocity is in the world frame
        wz = float((R @ d.qvel[3:6])[2])          # yaw rate = world-z component of the body angular velocity (an IMU
                                                  # attitude filter gives this; the raw body-z rate is not the yaw rate
                                                  # while the base pitches / rolls - that drifted 2.3 deg on one step)
        v = np.array([(1 + self.sx) * vb[0], (1 + self.sy) * vb[1]]) + self.rng.normal(0, self.noise_v, 2)
        w = (1 + self.sw) * wz + self.k_yaw * float(np.linalg.norm(vb[:2])) + self.rng.normal(0, self.noise_w)
        self.yaw = wrap(self.yaw + w * dt)
        self.p = self.p + rot2(self.yaw) @ v * dt

    # ------------------------------------------------------------------ frames
    def pose(self):
        return self.p.copy(), self.yaw

    def _T(self):
        dth = wrap(self.yaw - self.true_yaw)
        return dth, self.p - rot2(dth) @ self.true_p

    def to_est(self, P):
        dth, t = self._T()
        P = np.array(P, float)
        P[..., :2] = P[..., :2] @ rot2(dth).T + t
        return P

    def vec_to_est(self, V):
        dth, _ = self._T()
        V = np.array(V, float)
        V[..., :2] = V[..., :2] @ rot2(dth).T
        return V

    def to_true(self, P):
        dth, t = self._T()
        P = np.array(P, float)
        P[..., :2] = (P[..., :2] - t) @ rot2(dth)
        return P

    def vec_to_true(self, V):
        dth, _ = self._T()
        V = np.array(V, float)
        V[..., :2] = V[..., :2] @ rot2(dth)
        return V

    def error(self):
        """(position error [m], heading error [deg]) of the estimate vs the truth."""
        return float(np.linalg.norm(self.p - self.true_p)), float(np.degrees(wrap(self.yaw - self.true_yaw)))

    # ------------------------------------------------------------------ landmark correction
    def correct(self, obs_pos, obs_nxy, anchor_pos, anchor_nxy, gain=0.3, max_dp=0.6, max_dth=np.radians(15)):
        """A marker seen at obs (estimated frame, from the current estimate) is known to be at anchor: rotate +
        translate the robot's estimate so that the observation lands on the anchor. Returns (dp, dth) applied."""
        dth = wrap(np.arctan2(anchor_nxy[1], anchor_nxy[0]) - np.arctan2(obs_nxy[1], obs_nxy[0]))
        rel = np.asarray(obs_pos[:2], float) - self.p                      # robot -> marker, estimated frame
        p_new = np.asarray(anchor_pos[:2], float) - rot2(dth) @ rel        # where the robot must be
        dp = p_new - self.p
        if np.linalg.norm(dp) > max_dp or abs(dth) > max_dth:
            return None                                                    # outlier (bad PnP)
        self.p = self.p + gain * dp
        self.yaw = wrap(self.yaw + gain * dth)
        self.corrections.append((float(np.linalg.norm(dp)), float(np.degrees(dth))))
        return gain * dp, gain * dth
