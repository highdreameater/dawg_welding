"""Robot configurations for the stance WBC (quadruped + arm + welding torch), built from MuJoCo Menagerie models.

Every model is rebuilt so that *all* leg and arm joints are driven by pure torque motors (the WBC outputs torques).
"""
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MENAGERIE = os.path.join(ROOT, "third_party", "mujoco_menagerie")


@dataclass
class RobotCfg:
    name: str
    scene_xml: str
    base_body: str
    leg_joints: list            # 12 names, grouped per leg (order of feet below)
    foot_bodies: list           # body holding each foot sphere, same leg order
    foot_radius: float
    leg_q0: dict                # nominal stance (also the RL locomotion default pose -> clean hand-off)
    arm_joints: list
    arm_stow: np.ndarray
    tau_limits: dict            # joint -> |tau| max [Nm]
    torch_parent: str           # last arm link; torch is mounted along its +x (tool) axis
    torch_pos: float            # torch body origin along parent x [m]
    torch_tip: float            # tip distance from torch body origin [m]
    workpiece_x: float          # seam x (world) with robot base starting at x=0
    workpiece_half: tuple       # block half sizes
    ik_seeds: list
    arm_xml: str = None         # arm to attach (if not part of the robot model)
    arm_mount: tuple = None     # (parent body, pos)
    arm_prefix: str = ""
    brace_dz: float = 0.04
    sweep_amp: tuple = None     # stress test: per-arm-joint sweep amplitude [rad]
    wbc_gains: dict = field(default_factory=dict)  # StanceWBC kwargs overrides
    cam: dict = field(default_factory=lambda: dict(lookat=[0.5, 0, 0.3], distance=2.4, azimuth=140, elevation=-20))

    @property
    def foot_names(self):
        return ["foot_" + b for b in self.foot_bodies]

    def arm_names(self):
        return [self.arm_prefix + j for j in self.arm_joints]


def anymal_c_z1():
    legs = ["LF", "RF", "LH", "RH"]
    leg = [f"{l}_{j}" for l in legs for j in ("HAA", "HFE", "KFE")]
    # Isaac Lab ANYMAL_C_CFG default joint pos (what the pretrained rough-terrain policy stands in)
    q0 = {}
    for l in legs:
        front = l[1] == "F"
        q0[f"{l}_HAA"] = 0.0
        q0[f"{l}_HFE"] = 0.4 if front else -0.4
        q0[f"{l}_KFE"] = -0.8 if front else 0.8
    tau = {j: 80.0 for j in leg}  # ANYmal-C SEA drives (Menagerie forcerange / Isaac Lab effort limit)
    z1 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    tau.update({"z1_" + j: (60.0 if j == "joint2" else 30.0) for j in z1})  # Unitree Z1 limits (Menagerie)
    return RobotCfg(
        name="anymal_c_z1",
        scene_xml=os.path.join(MENAGERIE, "anybotics_anymal_c", "scene.xml"),
        base_body="base", leg_joints=leg,
        foot_bodies=[f"{l}_SHANK" for l in legs], foot_radius=0.03,
        leg_q0=q0,
        # Z1 zero pose = folded, but joint2/joint3 sit exactly on their limits there -> stow just inside them
        arm_joints=z1, arm_stow=np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0]),
        tau_limits=tau, torch_parent="z1_link06", torch_pos=0.06, torch_tip=0.21,
        workpiece_x=0.70, workpiece_half=(0.08, 0.30, 0.15),  # Z1 is short: narrow beam just ahead of the front feet
        ik_seeds=[[0, 1.2, -1.2, -0.3, 0, 0], [0, 1.6, -1.6, 0.2, 0, 0], [0, 0.9, -0.6, -0.8, 0, 0]],
        arm_xml=os.path.join(MENAGERIE, "unitree_z1", "z1.xml"), arm_mount=("base", (0.22, 0.0, 0.09)),
        arm_prefix="z1_", sweep_amp=(1.0, 0.35, 0.4, 0.5, 0, 0),
        # Z1 joints have 1 Nm Coulomb friction: stiffer torch task + integral action (tuned: 0.17 mm RMS, 68% peak torque)
        wbc_gains=dict(kp_ee=2500.0, kd_ee=100.0, ki_ee=8000.0),
        cam=dict(lookat=[0.4, 0, 0.35], distance=2.6, azimuth=140, elevation=-20),
    )


ROBOTS = {"anymal_c_z1": anymal_c_z1}  # Spot config lives in archive/spot/spot_robot_config.py


def build_model(cfg: RobotCfg, torch_mass=1.2, workpiece=True, render_size=(1280, 720)):
    spec = mujoco.MjSpec.from_file(cfg.scene_xml)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = render_size

    if cfg.arm_xml:  # mount the arm on the quadruped
        arm = mujoco.MjSpec.from_file(cfg.arm_xml)
        parent, pos = cfg.arm_mount
        frame = spec.body(parent).add_frame()
        frame.pos = list(pos)
        frame.attach_body(arm.worldbody.first_body(), cfg.arm_prefix, "")
        # the folded Z1 stow pose touches itself; arm self-collision is irrelevant for welding -> exclude
        arm_bodies = [b.name for b in spec.bodies if b.name.startswith(cfg.arm_prefix)]
        for i, b1 in enumerate(arm_bodies):
            for b2 in arm_bodies[i + 1:]:
                e = spec.add_exclude()
                e.bodyname1, e.bodyname2 = b1, b2

    # name the foot spheres
    for b in cfg.foot_bodies:
        spheres = [g for g in spec.body(b).geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
        assert len(spheres) == 1, f"expected one foot sphere in {b}, got {len(spheres)}"
        spheres[0].name = "foot_" + b

    # welding torch on the last arm link (non-colliding: welding is a non-contact process)
    tb = spec.body(cfg.torch_parent).add_body()
    tb.name, tb.pos = "torch", [cfg.torch_pos, 0, 0]
    neck = tb.add_geom()
    neck.type, neck.fromto, neck.size = mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0, cfg.torch_tip - 0.04, 0, 0], [0.022, 0, 0]
    neck.mass, neck.contype, neck.conaffinity, neck.rgba = torch_mass, 0, 0, [0.75, 0.2, 0.15, 1]
    nozzle = tb.add_geom()
    nozzle.type, nozzle.fromto, nozzle.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [cfg.torch_tip - 0.04, 0, 0, cfg.torch_tip - 0.005, 0, 0], [0.012, 0, 0]
    nozzle.mass, nozzle.contype, nozzle.conaffinity, nozzle.rgba = 0.01, 0, 0, [0.8, 0.7, 0.3, 1]
    tip = tb.add_site()
    tip.name, tip.pos, tip.size, tip.rgba = "torch_tip", [cfg.torch_tip, 0, 0], [0.006, 0, 0], [1, 0.4, 0, 1]

    if workpiece:
        hx, hy, hz = cfg.workpiece_half
        g = spec.worldbody.add_geom()
        g.name, g.type = "workpiece", mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.rgba = [hx, hy, hz], [cfg.workpiece_x, 0, hz], [0.45, 0.47, 0.5, 1]
        s = spec.worldbody.add_geom()
        s.type, s.size = mujoco.mjtGeom.mjGEOM_BOX, [0.002, 0.2, 0.0006]
        s.pos, s.rgba = [cfg.workpiece_x, 0, 2 * hz + 0.0006], [0.1, 0.1, 0.1, 1]
        s.contype = s.conaffinity = 0

    # replace every actuator by a torque motor with the robot's torque limit
    for a in list(spec.actuators):
        spec.delete(a)
    for j in cfg.leg_joints + cfg.arm_names():
        lim = cfg.tau_limits[j]
        a = spec.add_actuator()
        a.name, a.target, a.trntype = j, j, mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        a.gainprm = [1] + [0] * 9
        a.ctrlrange, a.forcerange = [-lim, lim], [-lim, lim]
        a.ctrllimited = a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    model = spec.compile()
    model.opt.timestep = 0.002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model
