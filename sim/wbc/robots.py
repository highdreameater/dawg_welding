"""Robot configurations for the stance WBC (Unitree Go2 + Z1 arm, optional welding torch), built from MuJoCo Menagerie models.

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
    seam_half: float = 0.15     # weld seam half-length along y [m]
    arm_armature: float = None  # rotor inertia added to arm joints [kg m^2] (Menagerie Z1 has none)
    cam: dict = field(default_factory=lambda: dict(lookat=[0.5, 0, 0.3], distance=2.4, azimuth=140, elevation=-20))
    pipe_work_dist: float = 0.56  # base centre -> force-control arc on the pipe (reach with a tilted torch) [m]
    leg_passive: tuple = None     # (damping, frictionloss) override for the leg joints, e.g. to match the RL training sim

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


def go2_z1_official():
    """Go2 + Z1 on the official Unitree mount (aliengoZ1 rule), no torch by default (massless tool point 5 cm past the
    flange). Used by sim/stance_demo.py, sim/terrain_demo.py, sim/myscene_demo.py, sim/aruco_mission_demo.py."""
    legs = ["FL", "FR", "RL", "RR"]
    leg = [f"{l}_{j}_joint" for l in legs for j in ("hip", "thigh", "calf")]
    # default pose of the pretrained Go2 policies (wty-yy MoE-CTS / symmetry) -> clean RL <-> WBC hand-off
    q0 = {}
    for l in legs:
        q0[f"{l}_hip_joint"] = 0.1 if l[1] == "L" else -0.1
        q0[f"{l}_thigh_joint"] = 0.8 if l[0] == "F" else 1.0
        q0[f"{l}_calf_joint"] = -1.5
    tau = {j: (45.43 if "calf" in j else 23.7) for j in leg}  # Go2 motor limits (Menagerie)
    z1 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    tau.update({"z1_" + j: (60.0 if j == "joint2" else 30.0) for j in z1})
    return RobotCfg(
        name="go2_z1_official",
        scene_xml=os.path.join(MENAGERIE, "unitree_go2", "scene.xml"),
        base_body="base", leg_joints=leg,
        foot_bodies=[f"{l}_calf" for l in legs], foot_radius=0.022,
        leg_q0=q0,
        arm_joints=z1, arm_stow=np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0]),
        # no torch: massless tool point 5 cm past the Z1 flange
        tau_limits=tau, torch_parent="z1_link06", torch_pos=0.051, torch_tip=0.05,
        # reach map (tool down, seam +-0.10 m): exact for work-surface height >= 0.10 m, x 0.35..0.55 m.
        # 12 cm-high beam at x = 0.42 m (front feet at x ~0.18-0.20 m)
        workpiece_x=0.42, workpiece_half=(0.06, 0.20, 0.06),
        ik_seeds=[[0, 1.2, -1.2, -0.3, 0, 0], [0, 1.6, -1.6, 0.2, 0, 0], [0, 0.9, -0.6, -0.8, 0, 0]],
        # official Unitree mount rule (aliengoZ1_description): x = trunk_length/2 - 0.07, z = trunk_height/2
        arm_xml=os.path.join(MENAGERIE, "unitree_z1", "z1.xml"), arm_mount=("base", (0.3762 / 2 - 0.07, 0.0, 0.114 / 2)),
        arm_prefix="z1_", sweep_amp=(1.0, 0.35, 0.4, 0.5, 0, 0), brace_dz=0.03, seam_half=0.10,
        # Without a torch the Z1 wrist link is ~inertia-free (0.29 kg, Menagerie armature 0) while its joint has 1 Nm
        # friction -> friction compensation chatters and the wrist roll runs into its limit. Give the Z1 joints a
        # realistic gearbox rotor inertia (same value Menagerie uses for the Go2 leg motors).
        arm_armature=0.01,
        # weld precision: arm disturbance observer (replaces fixed friction FF) while the tool is tracked in Cartesian
        # space -> 0.08 mm RMS tool error (vs 2.2 mm with friction FF only)
        wbc_gains=dict(kp_ee=2500.0, kd_ee=100.0, ki_ee=8000.0, dob_T=0.004, dob_cartesian_only=True),
        cam=dict(lookat=[0.25, 0, 0.2], distance=1.5, azimuth=140, elevation=-20),
    )


def go2_z1(mount_x=0.10):
    """Unitree Go2 with a Unitree Z1 on top of the base, 10 cm ahead of the base centre. The arm + torch (5.6 kg) is
    ~37% of the bare Go2's mass. Mounted over the centre the Z1 only reaches ~0.5 m ahead at pipe height (the Go2 is
    0.2 m lower than ANYmal-C); +10 cm buys ~0.1 m of reach for a ~3 cm forward shift of the CoM."""
    legs = ["FL", "FR", "RL", "RR"]
    leg = [f"{l}_{j}_joint" for l in legs for j in ("hip", "thigh", "calf")]
    # Isaac Lab UNITREE_GO2_CFG default joint pos (stance of the pretrained rough-terrain policy)
    q0 = {}
    for l in legs:
        q0[f"{l}_hip_joint"] = 0.1 if l[1] == "L" else -0.1
        q0[f"{l}_thigh_joint"] = 0.8 if l[0] == "F" else 1.0
        q0[f"{l}_calf_joint"] = -1.5
    tau = {j: (45.43 if "calf" in j else 23.7) for j in leg}  # Menagerie go2.xml / Go2 motor limits
    z1 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    tau.update({"z1_" + j: (60.0 if j == "joint2" else 30.0) for j in z1})
    return RobotCfg(
        name="go2_z1",
        scene_xml=os.path.join(MENAGERIE, "unitree_go2", "scene.xml"),
        base_body="base", leg_joints=leg,
        foot_bodies=[f"{l}_calf" for l in legs], foot_radius=0.022,
        leg_q0=q0,
        arm_joints=z1, arm_stow=np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0]),
        tau_limits=tau, torch_parent="z1_link06", torch_pos=0.06, torch_tip=0.21,
        workpiece_x=0.48, workpiece_half=(0.08, 0.30, 0.15),  # beam face clears the Go2 head (x 0.335)
        ik_seeds=[[0, 1.2, -1.2, -0.3, 0, 0], [0, 1.6, -1.6, 0.2, 0, 0], [0, 0.9, -0.6, -0.8, 0, 0]],
        arm_xml=os.path.join(MENAGERIE, "unitree_z1", "z1.xml"), arm_mount=("base", (mount_x, 0.0, 0.057)),
        pipe_work_dist=0.46,  # head (x 0.335) stops ~4 cm short of the pipe end; it cannot overhang the pipe
        # Isaac Lab's Go2 has no passive joint damping/friction (only the PD's kd = 0.5). Menagerie's damping 2 /
        # frictionloss 0.2 is 4x that damping and halves the pretrained policy's yaw tracking -> match Isaac Lab.
        leg_passive=(0.0, 0.0),
        arm_prefix="z1_", brace_dz=0.02, sweep_amp=(1.0, 0.35, 0.4, 0.5, 0, 0),
        wbc_gains=dict(kp_ee=2500.0, kd_ee=100.0, ki_ee=8000.0),
        cam=dict(lookat=[0.3, 0, 0.25], distance=1.8, azimuth=140, elevation=-20),
    )


# Robot + arm configs. To try another arm, add a config with a different arm_xml / arm_mount / arm_joints /
# tau_limits (e.g. a Menagerie arm fetched with menagerie models ready: anybotics_anymal_c unitree_z1 unitree_go2).
ROBOTS = {"anymal_c_z1": anymal_c_z1, "go2_z1": go2_z1, "go2_z1_official": go2_z1_official}


CONTACT_TIP_RADIUS = 0.008


def build_model(cfg: RobotCfg, torch_mass=1.2, workpiece=True, render_size=(1280, 720), world=None, contact_tip=False,
                scene_fn=None, cameras=None):
    """workpiece: the stand-alone beam of sim/stance_demo.py (ignored when `world` is given).
    world: a sim/world.py World (walls, obstacles, stations) added to the scene.
    contact_tip: a frictionless ball at the torch tip that touches the pipe (collision bit 2) - for force control.
    scene_fn(spec): optional callback that adds world geometry (stairs, walls, ...) before compiling.
    cameras: optional robot-mounted MuJoCo cameras, dicts (name, body, pos, quat, fovy) - e.g. the head RGB camera
             of sim/perception/aruco.py."""
    spec = mujoco.MjSpec.from_file(cfg.scene_xml)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = render_size
    for k in list(spec.keys):  # keyframes (e.g. Go2 "home") no longer match once the arm is attached
        spec.delete(k)

    if cfg.arm_xml:  # mount the arm on the quadruped
        arm = mujoco.MjSpec.from_file(cfg.arm_xml)
        parent, pos = cfg.arm_mount
        frame = spec.body(parent).add_frame()
        frame.pos = list(pos)
        frame.attach_body(arm.worldbody.first_body(), cfg.arm_prefix, "")
        # the folded Z1 stow pose touches itself; arm self-collision is irrelevant for welding -> exclude
        # (on small robots like Go2 the folded arm also rests on the trunk -> exclude arm-vs-base contacts too)
        arm_bodies = [b.name for b in spec.bodies if b.name.startswith(cfg.arm_prefix)]
        for i, b1 in enumerate(arm_bodies):
            for b2 in arm_bodies[i + 1:] + [cfg.base_body]:
                e = spec.add_exclude()
                e.bodyname1, e.bodyname2 = b1, b2

    # name the foot spheres
    for b in cfg.foot_bodies:
        spheres = [g for g in spec.body(b).geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
        assert len(spheres) == 1, f"expected one foot sphere in {b}, got {len(spheres)}"
        spheres[0].name = "foot_" + b

    # welding torch on the last arm link (non-colliding: welding is a non-contact process).
    # torch_mass <= 0: no torch body/mass, only a massless tool point ("torch_tip" site) torch_tip beyond torch_pos.
    tb = spec.body(cfg.torch_parent).add_body()
    tb.name, tb.pos = "torch", [cfg.torch_pos, 0, 0]
    if torch_mass > 0:
        neck = tb.add_geom()
        neck.type, neck.fromto, neck.size = mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0, cfg.torch_tip - 0.04, 0, 0], [0.022, 0, 0]
        neck.mass, neck.contype, neck.conaffinity, neck.rgba = torch_mass, 0, 0, [0.75, 0.2, 0.15, 1]
        nozzle = tb.add_geom()
        nozzle.type, nozzle.fromto, nozzle.size = mujoco.mjtGeom.mjGEOM_CYLINDER, [cfg.torch_tip - 0.04, 0, 0, cfg.torch_tip - 0.005, 0, 0], [0.012, 0, 0]
        nozzle.mass, nozzle.contype, nozzle.conaffinity, nozzle.rgba = 0.01, 0, 0, [0.8, 0.7, 0.3, 1]
    tip = tb.add_site()
    tip.name, tip.pos, tip.size, tip.rgba = "torch_tip", [cfg.torch_tip, 0, 0], [0.006, 0, 0], [1, 0.4, 0, 1]
    if contact_tip:
        # ball tip centred so that its surface is at the torch_tip site (the site is the contact point on the work)
        b = tb.add_geom()
        b.name, b.type, b.size = "contact_tip", mujoco.mjtGeom.mjGEOM_SPHERE, [CONTACT_TIP_RADIUS, 0, 0]
        b.pos, b.mass, b.group, b.rgba = [cfg.torch_tip - CONTACT_TIP_RADIUS, 0, 0], 0.005, 2, [0.9, 0.9, 0.95, 1]
        b.contype = b.conaffinity = 2   # touches only geoms with bit 2 (the pipe), not the robot / floor
        b.condim, b.priority = 1, 2     # frictionless normal contact (ball / roller tip)

    for c in cameras or []:
        cam = spec.body(c["body"]).add_camera()
        cam.name, cam.pos, cam.quat, cam.fovy = c["name"], list(c["pos"]), list(c["quat"]), c["fovy"]

    if world is not None:
        world.add_to(spec)
    elif workpiece:
        hx, hy, hz = cfg.workpiece_half
        g = spec.worldbody.add_geom()
        g.name, g.type = "workpiece", mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos, g.rgba = [hx, hy, hz], [cfg.workpiece_x, 0, hz], [0.45, 0.47, 0.5, 1]
        s = spec.worldbody.add_geom()
        s.type, s.size = mujoco.mjtGeom.mjGEOM_BOX, [0.002, 0.2, 0.0006]
        s.pos, s.rgba = [cfg.workpiece_x, 0, 2 * hz + 0.0006], [0.1, 0.1, 0.1, 1]
        s.contype = s.conaffinity = 0

    if scene_fn is not None:
        scene_fn(spec)

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
    if cfg.leg_passive is not None:
        for j in cfg.leg_joints:
            dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            model.dof_damping[dof], model.dof_frictionloss[dof] = cfg.leg_passive
    model.opt.timestep = 0.002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if cfg.arm_armature is not None:
        for j in cfg.arm_names():
            model.dof_armature[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] = cfg.arm_armature
    return model


def controller_model(cfg: RobotCfg, torch_mass=1.2):
    """The WBC's private "belief" model: robot + torch only (no world geoms) with contacts disabled. The controller
    only needs M, h and Jacobians; skipping collision detection makes its mj_forward much cheaper."""
    m = build_model(cfg, torch_mass=torch_mass, workpiece=False, contact_tip=True)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return m
