"""Archived: Spot + Spot arm config for the stance WBC (the project moved to ANYmal-C + Z1).

Produced archive/spot/results/stance_wbc/. To re-run it, register it in sim/wbc/robots.py:
    from archive.spot.spot_robot_config import spot_arm   (or copy the function)
    ROBOTS["spot"] = spot_arm
and fetch the model with `bash scripts/setup_assets.sh --spot`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim"))
from wbc.robots import MENAGERIE, RobotCfg  # noqa: E402


def spot_arm():
    leg = ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn", "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"]
    tau = {j: (115.0 if j.endswith("kn") else 45.0) for j in leg}
    tau.update({"arm_sh0": 90, "arm_sh1": 90, "arm_el0": 90, "arm_el1": 90, "arm_wr0": 30, "arm_wr1": 30, "arm_f1x": 30})
    return RobotCfg(
        name="spot",
        scene_xml=os.path.join(MENAGERIE, "boston_dynamics_spot", "scene_arm.xml"),
        base_body="body", leg_joints=leg,
        foot_bodies=["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"], foot_radius=0.036,
        leg_q0={"fl_hx": 0.1, "hl_hx": 0.1, "fr_hx": -0.1, "hr_hx": -0.1, "fl_hy": 0.9, "fr_hy": 0.9,
                "hl_hy": 1.1, "hr_hy": 1.1, "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5},
        arm_joints=["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"],
        arm_stow=np.array([0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0]),
        tau_limits=tau, torch_parent="arm_link_wr1", torch_pos=0.13, torch_tip=0.21,
        workpiece_x=0.95, workpiece_half=(0.15, 0.30, 0.09),
        ik_seeds=[[0, -0.8, 1.6, 0, 0.8, 0, 0], [-0.3, -0.6, 1.2, 0, 1.0, 0, 0], [0, -1.2, 2.0, 0, 0.6, 0, 0]],
        sweep_amp=(1.0, 0.35, 0.4, 0, 0.5, 0, 0),
    )
