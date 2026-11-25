"""Stance Whole-Body Controller (WBC) for a quadruped + arm + welding torch, in MuJoCo. Robot-agnostic (see robots.py).

Floating-base inverse-dynamics QP solved every physics tick (500 Hz):

    decision variables  x = [qdd (nv) , f (12) ]        f = 3-D ground-reaction force of each of the 4 feet

    min   w_base ||qdd_base - a_base*||^2                          (base pose held rigid: PD -> desired accel)
        + w_ee   ||J_ee qdd + Jd_ee qd - a_ee*||^2                  (torch tip follows the weld path, 6-D)
        + w_post ||qdd_arm - a_post*||^2                            (arm posture / joint-space trajectory)
        + eps_q ||qdd||^2 + eps_f ||f||^2
    s.t.  M_b qdd + h_b = Jc_b^T f                                  (unactuated floating-base rows of dynamics)
          Jc qdd + Jd_c qd = -K_c Jc qd                             (stance feet do not accelerate)
          |f_x|,|f_y| <= mu f_z ,  f_min <= f_z <= f_max             (linearised friction cone, unilateral)
          tau = M_a qdd + h_a - Jc_a^T f ,  |tau| <= tau_max          (actuator torque limits)
    with h = Coriolis/centrifugal + gravity - passive (joint damping/springs).

Dynamics (M, h, Jacobians, Jdot) come from MuJoCo itself (mj_fullM, qfrc_bias, mj_jac, mj_jacDot) evaluated on the
controller's *own* model copy, so the "belief" model can differ from the plant (e.g. wrong torch mass).
"""
import mujoco
import numpy as np
import qpsolvers

from .robots import RobotCfg


class Indexer:
    """Name -> qpos/dof/actuator index helpers."""

    def __init__(self, m, cfg: RobotCfg):
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        arm = cfg.arm_names()
        self.leg_q = np.array([m.jnt_qposadr[J(n)] for n in cfg.leg_joints])
        self.leg_v = np.array([m.jnt_dofadr[J(n)] for n in cfg.leg_joints])
        self.arm_q = np.array([m.jnt_qposadr[J(n)] for n in arm])
        self.arm_v = np.array([m.jnt_dofadr[J(n)] for n in arm])
        self.n_arm = len(arm)
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body)
        self.tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "torch_tip")
        self.feet = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in cfg.foot_names])
        self.foot_radius = cfg.foot_radius
        self.arm_lo = m.jnt_range[[J(n) for n in arm], 0]
        self.arm_hi = m.jnt_range[[J(n) for n in arm], 1]
        self.tau_max_dof = np.zeros(m.nv)
        for a in range(m.nu):
            self.tau_max_dof[self.act_dof[a]] = m.actuator_forcerange[a, 1]
        self.leg_q0 = np.array([cfg.leg_q0[n] for n in cfg.leg_joints])
        self.arm_stow = np.asarray(cfg.arm_stow, float)


def initial_state(m, d, ix: Indexer, xy=(0.0, 0.0), yaw=0.0):
    """Nominal stance with the arm stowed, base level at (x, y) facing `yaw`, feet exactly on the floor."""
    mujoco.mj_resetData(m, d)
    d.qpos[0:2] = xy
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[ix.leg_q] = ix.leg_q0
    d.qpos[ix.arm_q] = ix.arm_stow
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    foot_z = d.geom_xpos[ix.feet, 2].min() - ix.foot_radius
    d.qpos[2] -= foot_z - 0.0005
    mujoco.mj_forward(m, d)


# ----------------------------------------------------------------------------------------------- kinematics utils
def quat_err_local(q_des, q_cur):
    e = np.zeros(3)
    mujoco.mju_subQuat(e, q_des, q_cur)  # rotation (in q_cur's local frame) taking q_cur to q_des
    return e


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def site_quat(d, sid):
    return mat2quat(d.site_xmat[sid])


def axis_error(R_cur, R_des):
    """Rotation error that aligns only the tool x-axis (roll about the tool axis is free), world frame."""
    return np.cross(R_cur[:, 0], R_des[:, 0])


def ik_arm(m, qpos_init, ix: Indexer, p_des, R_des, seed=None, iters=300, axis_only=False):
    """Damped-least-squares IK on the first 6 arm joints (base + legs frozen). Returns arm q and residual [m].
    axis_only: only the tool x-axis direction is constrained (5-DoF task, e.g. a ball tip)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos_init
    if seed is not None:
        d.qpos[ix.arm_q] = seed
    q_des = mat2quat(R_des)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    cols = ix.arm_v[:6]
    for _ in range(iters):
        mujoco.mj_fwdPosition(m, d)
        ep = p_des - d.site_xpos[ix.tip]
        Rc = d.site_xmat[ix.tip].reshape(3, 3)
        er = axis_error(Rc, R_des) if axis_only else Rc @ quat_err_local(q_des, site_quat(d, ix.tip))
        e = np.concatenate([ep, 0.5 * er])
        if np.linalg.norm(e) < 1e-5:
            break
        mujoco.mj_jacSite(m, d, jp, jr, ix.tip)
        if axis_only:
            jr = (np.eye(3) - np.outer(Rc[:, 0], Rc[:, 0])) @ jr
        Jm = np.vstack([jp[:, cols], 0.5 * jr[:, cols]])
        dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e)
        if axis_only and np.abs(dq).max() > 0.2:
            dq *= 0.2 / np.abs(dq).max()
        d.qpos[ix.arm_q[:6]] = np.clip(d.qpos[ix.arm_q[:6]] + dq, ix.arm_lo[:6], ix.arm_hi[:6])
    mujoco.mj_fwdPosition(m, d)
    res = np.linalg.norm(p_des - d.site_xpos[ix.tip])
    return d.qpos[ix.arm_q].copy(), res


# ----------------------------------------------------------------------------------------------- the WBC
class StanceWBC:
    def __init__(self, ctrl_model, cfg: RobotCfg, mu=0.6, f_min=5.0, f_max=800.0, solver="daqp",
                 w_base=100.0, w_post=1.0, kp_base=900.0, kd_base=60.0,
                 kp_ee=900.0, kd_ee=60.0, ki_ee=0.0, kp_post=400.0, kd_post=40.0, kc=40.0,
                 kf=0.002, kv_n=60.0, vn_max=0.02, v_approach=0.015, f_contact=0.3, ff_ramp=0.2, ki_f=20.0,
                 dob_T=None, dob_cartesian_only=False):
        self.m = ctrl_model
        self.d = mujoco.MjData(ctrl_model)
        self.ix = Indexer(ctrl_model, cfg)
        self.nv = ctrl_model.nv
        self.n = self.nv + 12
        self.solver = solver
        self.w_base, self.w_post = w_base, w_post
        self.kp_base, self.kd_base, self.kp_ee, self.kd_ee = kp_base, kd_base, kp_ee, kd_ee
        self.kp_post, self.kd_post, self.kc = kp_post, kd_post, kc
        self.M = np.zeros((self.nv, self.nv))
        self.last_f = np.zeros(12)
        self.last_tau = np.zeros(self.nv)
        self.fail = 0
        self.friction = ctrl_model.dof_frictionloss.copy()  # Coulomb friction per dof (from the model)
        self.v_eps = 0.05                                   # [rad/s] smoothing of sign(v)
        self.ki_ee = ki_ee
        self.ee_int = np.zeros(6)
        self.ee_int_max = 0.002                             # [m*s] / [rad*s] anti-windup clamp
        self.dob_cartesian_only = dob_cartesian_only
        self.dob_T = dob_T                                  # arm disturbance-observer filter time constant [s] (None: off)
        self.dist_hat = np.zeros(len(self.ix.arm_v))
        self.prev_qvel, self.prev_tau_arm = None, None
        # force control along the surface normal:
        #   free space (F_meas < f_contact): approach along -n at v_approach
        #   in contact: admittance vn_des = -kf (F_des - F_meas), |vn_des| <= vn_max, plus a feed-forward of the
        #   desired contact force in the dynamics, ramped in over ff_ramp seconds after touch-down, plus an
        #   integral of the force error (explicit PI force control: removes the steady-state force error)
        self.kf, self.kv_n, self.vn_max = kf, kv_n, vn_max
        self.v_approach, self.f_contact, self.ff_ramp = v_approach, f_contact, ff_ramp
        self.ki_f, self.f_int = ki_f, 0.0
        self.vn_des, self.f_ff = 0.0, 0.0
        # friction pyramid rows (per foot): fx-mu fz, -fx-mu fz, fy-mu fz, -fy-mu fz, -fz (<= -fmin), fz (<= fmax)
        G = []
        for k in range(4):
            for r in ([1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu], [0, 0, -1], [0, 0, 1]):
                row = np.zeros(12)
                row[3 * k:3 * k + 3] = r
                G.append(row)
        self.G_f = np.array(G)
        self.h_f = np.tile([0, 0, 0, 0, -f_min, f_max], 4).astype(float)

    def compute(self, qpos, qvel, ref):
        """ref: base_pos, base_quat, arm_q, arm_qd, arm_qdd, w_post, and optionally w_ee, ee_pos, ee_vel, ee_acc, ee_quat.
        Returns joint torques (nv, base rows ~0) and the planned foot forces (12)."""
        m, d, ix, nv = self.m, self.d, self.ix, self.nv
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, self.M)
        M, h = self.M, d.qfrc_bias - d.qfrc_passive

        # --- contact Jacobians at the bottom of each foot sphere
        Jc, Jdc = np.zeros((12, nv)), np.zeros((12, nv))
        jp = np.zeros((3, nv))
        for k, g in enumerate(ix.feet):
            p = d.geom_xpos[g] - np.array([0, 0, ix.foot_radius])
            b = m.geom_bodyid[g]
            mujoco.mj_jac(m, d, jp, None, p, b); Jc[3 * k:3 * k + 3] = jp
            mujoco.mj_jacDot(m, d, jp, None, p, b); Jdc[3 * k:3 * k + 3] = jp
        vc = Jc @ qvel

        A_cost, b_cost, w_cost = [], [], []
        # --- base task (free-joint dofs: lin acc in world frame, ang acc in body frame)
        a_lin = self.kp_base * (ref["base_pos"] - qpos[0:3]) - self.kd_base * qvel[0:3]
        a_ang = self.kp_base * quat_err_local(ref["base_quat"], qpos[3:7]) - self.kd_base * qvel[3:6]
        Ab = np.zeros((6, self.n)); Ab[:, :6] = np.eye(6)
        A_cost.append(Ab); b_cost.append(np.concatenate([a_lin, a_ang])); w_cost.append(self.w_base)

        # --- torch-tip task (6-D, world frame)
        w_ee = ref.get("w_ee", 0.0)
        if w_ee > 0:
            jpe, jre = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jacSite(m, d, jpe, jre, ix.tip)
            Je = np.vstack([jpe, jre])
            jpd, jrd = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jacDot(m, d, jpd, jrd, d.site_xpos[ix.tip], m.site_bodyid[ix.tip])
            Jde = np.vstack([jpd, jrd])
            Rc = d.site_xmat[ix.tip].reshape(3, 3)
            if ref.get("axis_only", False):
                # only the tool axis direction is controlled (ball tip: roll about the axis is free) -> project the
                # angular rows onto the plane normal to the tool axis
                Pa = np.eye(3) - np.outer(Rc[:, 0], Rc[:, 0])
                Je[3:], Jde[3:] = Pa @ Je[3:], Pa @ Jde[3:]
                R_des = np.zeros(9)
                mujoco.mju_quat2Mat(R_des, ref["ee_quat"])
                e_rot = axis_error(Rc, R_des.reshape(3, 3))
            else:
                e_rot = Rc @ quat_err_local(ref["ee_quat"], site_quat(d, ix.tip))
            v_ee = Je @ qvel
            e_pos = ref["ee_pos"] - d.site_xpos[ix.tip]
            fc = ref.get("force")
            if fc is not None:
                # hybrid force/motion: position control in the tangent plane, force control along the normal n
                # (n = outward surface normal; the surface pushes the tip with f_meas * n).
                n = np.asarray(fc["n"], float)
                P = np.eye(3) - np.outer(n, n)
                e_pos = P @ e_pos
                f_des, f_meas = fc["f_des"], fc["f_meas"]
                in_contact = f_meas > self.f_contact
                if in_contact or self.f_ff > 0:
                    vn_des = float(np.clip(-self.kf * (f_des - f_meas), -self.vn_max, self.vn_max))
                else:
                    vn_des = -self.v_approach
                a_n = n @ ref["ee_acc"] + self.kv_n * (vn_des - n @ v_ee[:3])  # curvature feed-forward + admittance
                a_lin = P @ (ref["ee_acc"] + self.kd_ee * (ref["ee_vel"] - v_ee[:3])) + self.kp_ee * e_pos + a_n * n
                # the desired contact force is an external force on the robot (surface pushes the tip with f n): it
                # enters the floating-base dynamics, so the stance feet / base absorb the reaction. Feeding the
                # *measured* force here instead would be positive feedback (soft contact force = applied push).
                step = f_des * m.opt.timestep / self.ff_ramp
                self.f_ff += float(np.clip((f_des if in_contact else 0.0) - self.f_ff, -step, step))
                if in_contact:
                    self.f_int = float(np.clip(self.f_int + self.ki_f * (f_des - f_meas) * m.opt.timestep, -f_des, f_des))
                else:
                    self.f_int *= 0.99  # contact lost: bleed the integral off
                h = h - jpe.T @ ((self.f_ff + self.f_int) * n)
                self.vn_des = vn_des
                # the normal direction is force-controlled: drop any position integral along n (e.g. from hover)
                self.ee_int[:3] = P @ self.ee_int[:3]
            else:
                self.f_ff = self.f_int = 0.0
                a_lin = ref["ee_acc"] + self.kp_ee * e_pos + self.kd_ee * (ref["ee_vel"] - v_ee[:3])
            # integral action (with clamp) removes residual friction / model-mismatch offsets at the torch tip
            self.ee_int = np.clip(self.ee_int + np.concatenate([e_pos, e_rot]) * m.opt.timestep, -self.ee_int_max, self.ee_int_max)
            a_des = np.concatenate([a_lin, self.kp_ee * e_rot - self.kd_ee * v_ee[3:]]) + self.ki_ee * self.ee_int
            Ae = np.zeros((6, self.n)); Ae[:, :nv] = Je
            A_cost.append(Ae); b_cost.append(a_des - Jde @ qvel); w_cost.append(w_ee)
        else:
            self.ee_int[:] = 0.0

        # --- expected contact force at the tool tip in joint-posture mode (ref["f_ext"]: world force the surface exerts
        # on the tip, e.g. F_des * n). It enters the dynamics like the force-mode feed-forward above, so the joints
        # supply the push by torque instead of yielding (deflection under an unmodelled 5 N slid the tip ~6 cm and
        # tilted the tool ~5 deg). Must be the *desired* force - the measured one would be positive feedback.
        f_ext = ref.get("f_ext")
        if f_ext is not None and w_ee == 0:
            jpt = np.zeros((3, nv))
            mujoco.mj_jacSite(m, d, jpt, None, ix.tip)
            h = h - jpt.T @ np.asarray(f_ext, float)

        # --- arm posture / joint trajectory
        na_ = ix.n_arm
        a_post = ref["arm_qdd"] + self.kp_post * (ref["arm_q"] - qpos[ix.arm_q]) + self.kd_post * (ref["arm_qd"] - qvel[ix.arm_v])
        Ap = np.zeros((na_, self.n)); Ap[np.arange(na_), ix.arm_v] = 1
        A_cost.append(Ap); b_cost.append(a_post); w_cost.append(ref.get("w_post", self.w_post))

        # --- leg posture (very weak; legs are already pinned by base + feet)
        Al = np.zeros((12, self.n)); Al[np.arange(12), ix.leg_v] = 1
        A_cost.append(Al); b_cost.append(-10.0 * qvel[ix.leg_v]); w_cost.append(1e-3)

        P = 1e-6 * np.eye(self.n)
        P[nv:, nv:] += 1e-5 * np.eye(12)
        q = np.zeros(self.n)
        for A, b, w in zip(A_cost, b_cost, w_cost):
            P += w * A.T @ A
            q -= w * A.T @ b

        # --- equalities: floating-base dynamics + stance feet
        Aeq = np.zeros((18, self.n)); beq = np.zeros(18)
        Aeq[:6, :nv] = M[:6]; Aeq[:6, nv:] = -Jc[:, :6].T; beq[:6] = -h[:6]
        Aeq[6:, :nv] = Jc; beq[6:] = -Jdc @ qvel - self.kc * vc

        # --- inequalities: friction cones + torque limits
        na = nv - 6
        Gt = np.zeros((2 * na, self.n))
        Gt[:na, :nv] = M[6:]; Gt[:na, nv:] = -Jc[:, 6:].T
        Gt[na:] = -Gt[:na]
        tmax = ix.tau_max_dof[6:]
        ht = np.concatenate([tmax - h[6:], tmax + h[6:]])
        Gf = np.zeros((24, self.n)); Gf[:, nv:] = self.G_f
        G = np.vstack([Gf, Gt]); hh = np.concatenate([self.h_f, ht])

        x = qpsolvers.solve_qp(P, q, G, hh, Aeq, beq, solver=self.solver)
        if x is None:  # infeasible -> keep last torques
            self.fail += 1
            return self.last_tau.copy(), self.last_f.copy()
        qdd, f = x[:nv], x[nv:]
        tau = M @ qdd + h - Jc.T @ f
        # dry-friction (frictionloss) feed-forward on the velocity the QP is commanding for the next tick.
        # Z1 joints have 1 Nm of Coulomb friction; uncompensated it causes stick-slip (~3.5 mm torch error).
        # Direction = *planned* joint velocity (clean sign through weave reversals), falling back to the commanded one.
        v_cmd = qvel + qdd * m.opt.timestep
        if w_ee > 0:
            Ja = Je[:, ix.arm_v]
            v_tip = np.array(ref["ee_vel"], float)
            if "force" in ref:  # force mode: the planned normal motion is the admittance velocity
                n = np.asarray(ref["force"]["n"], float)
                v_tip += (self.vn_des - n @ v_tip) * n
            # planned tool twist: the reference linear velocity *and* angular velocity. Without the angular part
            # (tool turning to follow a curved surface) the wrist joints' planned velocities - hence the friction
            # feed-forward signs - are wrong, and the feed-forward adds friction instead of cancelling it.
            w_tool = np.asarray(ref.get("ee_omega", np.zeros(3)), float)
            v_cmd[ix.arm_v] = np.linalg.pinv(Ja, rcond=1e-3) @ np.concatenate([v_tip, w_tool])
        else:
            v_cmd[ix.arm_v] = ref["arm_qd"]
        ff = self.friction[6:] * np.tanh(v_cmd[6:] / self.v_eps)
        if self.dob_T is not None:
            # arm disturbance observer: tau_dist = M qdd_meas + h - tau_applied (arm rows have no contact terms),
            # low-pass filtered. Always updated (estimate stays warm); applied *instead of* the fixed friction
            # feed-forward only while the tool is tracked in Cartesian space (welding precision mode).
            ia = ix.arm_v
            if self.prev_qvel is not None:
                qdd_meas = (qvel - self.prev_qvel) / m.opt.timestep
                resid = M[ia] @ qdd_meas + h[ia] - self.prev_tau_arm
                a = m.opt.timestep / (self.dob_T + m.opt.timestep)
                self.dist_hat += a * (resid - self.dist_hat)
            if w_ee > 0 or not self.dob_cartesian_only:
                tau[ia] -= self.dist_hat
                ff[ia - 6] = 0.0
        tau[6:] += ff
        tau[6:] = np.clip(tau[6:], -ix.tau_max_dof[6:], ix.tau_max_dof[6:])
        self.prev_qvel, self.prev_tau_arm = qvel.copy(), tau[ix.arm_v].copy()
        self.last_tau, self.last_f = tau, f
        return tau, f
