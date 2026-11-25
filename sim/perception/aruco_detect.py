"""Head RGB camera + OpenCV ArUco detection + PnP: find work-order markers in what the robot can actually see.

    camera   : 1280 x 720, 50 deg vertical FOV (~79 deg horizontal), on the Go2 head next to the depth camera,
               level (tags sit about at its height: pitched up, the plate's lower edge left the view at 0.3 m). Rendered by MuJoCo (occlusion, lighting, perspective are all real).
    detection: cv2.aruco.ArucoDetector (DICT_4X4_50), sub-pixel corners
    pose     : cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE) with the known marker edge (0.20 m) and the pinhole intrinsics
               of the MuJoCo camera (a planar marker seen face-on has two poses that fit about equally - the
               orientation of such a sighting is flagged unreliable, its position is still good)
               -> marker pose in the camera frame -> world frame via the robot's own pose
               (the map is known and the robot is localised in it - no SLAM; marker poses are *measured*).
    fusion   : per marker id, observations weighted by 1/distance^2 (closer views are more accurate); a marker is
               only accepted after 2 consistent sightings, with its plate seen >= 24 px wide. Confirmed markers are
               landmarks: later sightings re-localise the robot's drifting odometry (sim/perception/odometry.py).
    frames   : detect() returns poses in the physics frame of the camera pose it is given; the mission maps them
               into the robot's estimated frame (the camera is on the robot: only the relative pose is measured).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

CAM_NAME = "head_rgb"
W, H, FOVY = 1280, 720, 50.0
MOUNT = (0.31, 0.0, 0.075)        # in the base frame, just above the depth camera
PITCH_UP_DEG = 0.0                # level: sees the tag face-on at the work stance (0.3 m) as well as far away


def add_head_camera(spec, base_body="base"):
    """scene_fn for build_model: adds the head RGB camera to the robot base."""
    a = np.radians(PITCH_UP_DEG)
    cam = spec.body(base_body).add_camera()
    cam.name, cam.pos, cam.fovy = CAM_NAME, list(MOUNT), FOVY
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [0, -1, 0, -np.sin(a), 0, np.cos(a)]   # camera looks along -z = base +x pitched up by a
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, H)


@dataclass
class MarkerObs:
    id: int
    pos: np.ndarray          # world, marker centre
    R: np.ndarray            # world, columns = marker x (right), y (up), z (out of the plate)
    dist: float
    px: float                # apparent edge length [px]
    reproj_px: float
    t: float
    rot_ok: bool = True      # False: planar-PnP ambiguity (two poses fit the corners) -> trust the position only


@dataclass
class MarkerEstimate:
    id: int
    obs: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.obs)

    def fused(self):
        w = np.array([1.0 / max(o.dist, 0.3) ** 2 for o in self.obs])
        p = np.sum([wi * o.pos for wi, o in zip(w, self.obs)], 0) / w.sum()
        ro = [(wi, o) for wi, o in zip(w, self.obs) if o.rot_ok] or list(zip(w, self.obs))   # orientation: unambiguous views
        z = np.sum([wi * o.R[:, 2] for wi, o in ro], 0)
        z /= np.linalg.norm(z)
        y = np.sum([wi * o.R[:, 1] for wi, o in ro], 0)
        y -= (y @ z) * z
        y /= np.linalg.norm(y)
        return p, np.stack([np.cross(y, z), y, z], 1)

    @property
    def normal_xy(self):
        _, R = self.fused()
        n = R[:2, 2]
        return n / np.linalg.norm(n)


class ArucoCamera:
    def __init__(self, m, marker_size, dictionary="DICT_4X4_50", min_px=24.0, max_dist=7.0):
        import cv2
        self.cv2 = cv2
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        assert self.cam_id >= 0, "build the model with scene_fn=add_head_camera"
        self.renderer = mujoco.Renderer(m, H, W)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 1, 0, 0]
        D = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(D, p)
        f = (H / 2) / np.tan(np.radians(m.cam_fovy[self.cam_id]) / 2)
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        s = marker_size / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        self.min_px, self.max_dist = min_px, max_dist
        self.last_image = None

    def capture(self, m, d):
        self.renderer.update_scene(d, camera=self.cam_id, scene_option=self.opt)
        img = self.renderer.render()
        self.last_image = img
        return img

    def detect(self, m, d, t=None):
        """Render the head camera and return the list of MarkerObs (world frame)."""
        cv2 = self.cv2
        img = self.capture(m, d)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        c_world = d.cam_xpos[self.cam_id].copy()
        R_cam = d.cam_xmat[self.cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # OpenCV camera -> world
        for c, mid in zip(corners, ids.ravel()):
            c = c.reshape(4, 2).astype(np.float64)
            px = float(np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)))
            if px < self.min_px:
                continue
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, c, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not n_sol:
                continue
            errs = np.asarray(errs, float).ravel()
            k = int(np.argmin(errs))
            rvec, tvec, reproj = rvecs[k], tvecs[k], float(errs[k])
            Rm, _ = cv2.Rodrigues(rvec)
            rot_ok = True
            if n_sol > 1:                         # planar ambiguity: both poses explain the corners about equally
                R2, _ = cv2.Rodrigues(rvecs[1 - k])
                differ = np.degrees(np.arccos(np.clip(Rm[:, 2] @ R2[:, 2], -1, 1))) > 2.0
                rot_ok = not differ or errs[1 - k] > 4.0 * max(errs[k], 0.05)
            dist = float(np.linalg.norm(tvec))
            if dist > self.max_dist or reproj > 2.0:
                continue
            out.append(MarkerObs(int(mid), c_world + R_cam @ tvec.ravel(), R_cam @ Rm, dist, px, reproj,
                                 d.time if t is None else t, rot_ok))
        return out


class MarkerMap:
    """Work-order markers as landmarks, in the robot's estimated frame.

    update(obs) -> (newly confirmed ids, re-localisation pairs [(obs, anchor_pos, anchor_normal_xy)])
      - a marker is confirmed after `min_obs` consistent sightings
      - its anchor (fused pose, 1/d^2 weights) is refined only by views closer than 0.8 x the closest so far
      - every other sighting within `reloc_range` is a re-localisation measurement for the odometry"""

    def __init__(self, known_ids, min_obs=2, gate=0.25, reloc_range=3.5):
        self.known_ids, self.min_obs, self.gate, self.reloc_range = set(known_ids), min_obs, gate, reloc_range
        self.est = {}
        self.min_dist = {}
        self.confirmed = []

    def update(self, observations):
        new, reloc = [], []
        for o in observations:
            if o.id not in self.known_ids:
                continue                        # not one of our work orders
            e = self.est.setdefault(o.id, MarkerEstimate(o.id))
            if o.id not in self.confirmed:
                if e.n and np.linalg.norm(e.fused()[0] - o.pos) > self.gate + 0.05 * o.dist:
                    continue                    # inconsistent sighting (bad PnP flip etc.)
                e.obs.append(o)
                self.min_dist[o.id] = min(self.min_dist.get(o.id, np.inf), o.dist)
                if e.n >= self.min_obs:
                    self.confirmed.append(o.id)
                    new.append(o.id)
                continue
            if o.dist < 0.8 * self.min_dist[o.id]:
                if np.linalg.norm(e.fused()[0] - o.pos) < self.gate + 0.05 * o.dist:
                    e.obs.append(o)             # closer view: refine the anchor
                    self.min_dist[o.id] = o.dist
                continue
            if o.dist < self.reloc_range:
                p, R = e.fused()
                n = R[:2, 2] / np.linalg.norm(R[:2, 2])
                reloc.append((o, p, n))
        return new, reloc

    def pose(self, mid):
        return self.est[mid].fused()
