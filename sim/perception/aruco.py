"""Head RGB camera + ArUco detection and marker pose estimation (OpenCV), in world coordinates.

Camera: an RGB camera on the robot's head, Intel RealSense D435 colour-stream class (1280 x 720, 69 x 42.5 deg),
pitched 8 deg down. It is a MuJoCo camera on the base body (added by wbc.robots.build_model(cameras=...)) and is
rendered with the robot's own geoms (groups 2/3) hidden. Optional seeded Gaussian pixel noise.

Detection: cv2.aruco (DICT_4X4_50, sub-pixel corners) -> IPPE-square PnP with the known marker size -> pose in the
camera frame -> world frame with the camera pose (robot localisation is taken from the simulator, as everywhere in
this repo: there is no SLAM). The marker sits on a curved surface, so its corners are not coplanar with its centre:
the PnP plane passes through the corners, a few mm behind the centre point (reported, not corrected).

    cam = RgbCamera(model)                          # model built with cameras=[camera_spec(cfg.name)]
    det = ArucoDetector(cam.K, marker_size=0.12)
    rgb, cam_pos, cam_R = cam.capture(model, data)
    for obs in det.detect(rgb, cam_pos, cam_R, t=data.time): ...
"""
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

CAMERA_NAME = "head_rgb"
# mount on the base body: position in the base frame [m], pitch down [deg]
RGB_MOUNTS = {"go2_z1": ((0.30, 0.0, 0.05), 8.0), "anymal_c_z1": ((0.42, 0.0, 0.08), 8.0)}
CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])   # OpenCV camera axes (x right, y down, z forward) vs MuJoCo (x, y up, -z fwd)


def camera_spec(robot, fovy_deg=42.5):
    """Camera dict for wbc.robots.build_model(cameras=[...])."""
    pos, pitch = RGB_MOUNTS[robot]
    p = np.radians(pitch)
    fwd, right, up = np.array([np.cos(p), 0, -np.sin(p)]), np.array([0, -1.0, 0]), np.array([np.sin(p), 0, np.cos(p)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([right, up, -fwd], 1).ravel())
    return dict(name=CAMERA_NAME, body="base", pos=pos, quat=q, fovy=fovy_deg)


class RgbCamera:
    def __init__(self, m, name=CAMERA_NAME, width=1280, height=720, noise_std=2.0, seed=0):
        self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cid < 0:
            raise ValueError(f"model has no camera {name!r} (build it with cameras=[camera_spec(robot)])")
        self.name, self.width, self.height = name, width, height
        f = 0.5 * height / np.tan(np.radians(m.cam_fovy[self.cid]) / 2)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        self.hfov = 2 * np.arctan(0.5 * width / f)
        self.renderer = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = self.opt.geomgroup[3] = 0        # the robot does not see itself
        self.noise_std, self.rng = noise_std, np.random.default_rng(seed)

    def pose(self, d):
        return d.cam_xpos[self.cid].copy(), d.cam_xmat[self.cid].reshape(3, 3).copy()

    def capture(self, m, d):
        """(rgb image (H, W, 3) uint8, camera position, camera rotation (MuJoCo convention))."""
        self.renderer.update_scene(d, camera=self.cid, scene_option=self.opt)
        img = self.renderer.render()
        if self.noise_std > 0:
            img = np.clip(img + self.rng.normal(0, self.noise_std, img.shape), 0, 255).astype(np.uint8)
        return img, *self.pose(d)


@dataclass
class MarkerObservation:
    id: int
    p: np.ndarray              # marker centre, world [m]
    n: np.ndarray              # marker normal (out of the printed face), world
    dist: float                # camera -> marker [m]
    view_deg: float            # angle between the normal and the direction to the camera
    side_px: float             # mean side length in the image
    reproj_px: float           # reprojection error of the chosen PnP solution
    ambiguity: float           # reprojection error ratio best / second (IPPE): ~1 = the normal is ambiguous
    t: float                   # sim time of the image
    cam_pos: np.ndarray
    corners: np.ndarray = field(repr=False, default=None)   # (4, 2) image corners


class ArucoDetector:
    def __init__(self, K, marker_size=0.12, sizes=None, min_side_px=14.0, max_view_deg=75.0):
        from markers import dictionary
        self.K = np.asarray(K, float)
        self.size, self.sizes = marker_size, dict(sizes or {})   # per-id sizes if markers differ
        self.min_side_px, self.max_view_deg = min_side_px, max_view_deg
        prm = cv2.aruco.DetectorParameters()
        prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(dictionary(), prm)

    def _object_points(self, mid):
        h = 0.5 * self.sizes.get(mid, self.size)
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)   # TL TR BR BL

    def detect(self, rgb, cam_pos, cam_R, t=0.0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, mid in zip(corners, ids.ravel()):
            c = c.reshape(4, 2).astype(np.float64)
            side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)))
            if side < self.min_side_px:
                continue
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(self._object_points(int(mid)), c, self.K, None,
                                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not n_sol:
                continue
            errs = np.asarray(errs, float).ravel()
            k = int(np.argmin(errs))
            amb = float(errs[k] / errs[1 - k]) if n_sol > 1 and errs[1 - k] > 0 else 0.0
            R_m, _ = cv2.Rodrigues(rvecs[k])
            p = cam_pos + cam_R @ (CV_FROM_MJ @ np.asarray(tvecs[k], float).ravel())
            n = cam_R @ (CV_FROM_MJ @ R_m[:, 2])
            to_cam = cam_pos - p
            dist = float(np.linalg.norm(to_cam))
            view = float(np.degrees(np.arccos(np.clip(n @ to_cam / dist, -1, 1))))
            if view > self.max_view_deg:
                continue
            out.append(MarkerObservation(int(mid), p, n / np.linalg.norm(n), dist, view, side, float(errs[k]), amb,
                                         float(t), cam_pos.copy(), c))
        return out


def draw_detections(rgb, observations, scale=1.0):
    """Copy of the image with marker outlines and ids (for videos)."""
    img = rgb.copy()
    for o in observations:
        pts = (o.corners * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (40, 255, 40), max(1, int(3 * scale)))
        cv2.putText(img, f"id {o.id}  {o.dist:.1f} m", tuple(int(x) for x in pts[0, 0] + [0, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * max(scale, 0.5), (40, 255, 40), 2)
    return img
