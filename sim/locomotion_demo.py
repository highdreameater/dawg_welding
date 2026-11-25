"""Zero-shot MuJoCo demos of pretrained Unitree Go2 locomotion policies (base quadruped only, no arm).

Robot: MuJoCo Menagerie unitree_go2. Joint damping/friction set to each policy's training values, PD torques computed
at the physics rate (dt 0.005 s, policy at 50 Hz), torques clipped to Go2 limits (hip/thigh 23.7 Nm, calf 45.4 Nm).

Policies (policies/go2/):
  moe_cts    wty-yy go2_rl_lab  MoE-CTS 82.5k (RoboGauge 0.693) - blind, 45-D obs x 10 history (inside the TorchScript)
  symmetry   wty-yy go2_rl_robotlab symmetry v1 77k (RoboGauge 0.701) - same interface, cmd scale 1.0

    python sim/locomotion_demo.py                                  # both policies x {walk, turn, stairs} -> results/locomotion/
    python sim/locomotion_demo.py --arm                            # with the Z1 mounted (folded, static payload)
    python sim/locomotion_demo.py --policy moe_cts --scenario stairs --step-height 0.12
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GO2_XML = os.path.join(ROOT, "third_party", "mujoco_menagerie", "unitree_go2", "scene.xml")
JOINTS = [f"{l}_{j}_joint" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]  # policy order
Q0 = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
DT, DECIM = 0.005, 4
Z1_XML = os.path.join(ROOT, "third_party", "mujoco_menagerie", "unitree_z1", "z1.xml")
# official Unitree mount rule (aliengoZ1_description): x = trunk_length/2 - 0.07, z = trunk_height/2  (Go2 trunk 0.3762 x 0.114)
Z1_MOUNT = (0.3762 / 2 - 0.07, 0.0, 0.114 / 2)
Z1_STOW = np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0])  # folded (zero pose sits on the joint2/joint3 limits)

POLICIES = {
    "moe_cts": dict(file="moe_cts_82.5k/exported/policy.pt", kind="internal_hist",
                    cmd_scale=[0.5, 1.0, 1.0], kp=25.0, kd=0.5, act_scale=np.full(12, 0.25), friction=0.01,
                    label="wty-yy go2_rl_lab MoE-CTS 82.5k"),
    "symmetry": dict(file="symmetry_77k/exported/policy.pt", kind="internal_hist",
                     cmd_scale=[1.0, 1.0, 1.0], kp=25.0, kd=0.5, act_scale=np.full(12, 0.25), friction=0.01,
                     label="wty-yy robot_lab symmetry 77k"),
}


def command(scn, t):
    if scn == "stairs":
        return np.array([0.5, 0.0, 0.0]) if t > 1.0 else np.zeros(3)
    if scn == "turn":  # turn in place both ways, then walk circles both ways
        sched = [(1.0, (0, 0, 0)), (4.5, (0, 0, 1.0)), (8.0, (0, 0, -1.0)), (12.0, (0.6, 0, 0.8)),
                 (16.0, (0.6, 0, -0.8)), (1e9, (0, 0, 0))]
        for t_end, c in sched:
            if t < t_end:
                return np.array(c, float)
    for t_end, c in [(1.0, (0, 0, 0)), (5.0, (0.8, 0, 0)), (8.0, (0.5, 0, 0.8)), (11.0, (0.8, 0, -0.8)),
                     (13.5, (-0.5, 0, 0)), (1e9, (0, 0, 0))]:
        if t < t_end:
            return np.array(c, float)


def build(scn, step_h, friction, arm=False):
    spec = mujoco.MjSpec.from_file(GO2_XML)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    if arm:  # Unitree Z1 (Menagerie, keeps its own position servos) rigidly mounted on the Go2 trunk
        z1 = mujoco.MjSpec.from_file(Z1_XML)
        frame = spec.body("base").add_frame()
        frame.pos = list(Z1_MOUNT)
        frame.attach_body(z1.worldbody.first_body(), "z1_", "")
        arm_bodies = [b.name for b in spec.bodies if b.name.startswith("z1_")]
        for i, b1 in enumerate(arm_bodies):  # folded arm rests on itself / the trunk: no internal contacts
            for b2 in arm_bodies[i + 1:] + ["base"]:
                e = spec.add_exclude()
                e.bodyname1, e.bodyname2 = b1, b2
    if scn == "stairs":
        run, n, x0 = 0.30, 6, 1.0
        def box(cx, h, half_x):
            g = spec.worldbody.add_geom()
            g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_BOX, [half_x, 1.0, h / 2], [cx, 0, h / 2], [0.6, 0.6, 0.65, 1]
        for i in range(n):  # up
            box(x0 + run * i + run / 2, step_h * (i + 1), run / 2)
        top = x0 + run * n
        box(top + 0.6, step_h * n, 0.6)  # platform 1.2 m
        for i in range(n):  # down
            box(top + 1.2 + run * i + run / 2, step_h * (n - i), run / 2)
    m = spec.compile()
    m.opt.timestep = DT
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # stable stiff Z1 servos; same for arm / no-arm runs
    for j in JOINTS:  # match training joint dynamics (Menagerie default: damping 2, frictionloss 0.2)
        dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        m.dof_damping[dof], m.dof_frictionloss[dof] = 0.0, friction
    return m


class Runner:
    def __init__(self, name, scn, step_h, arm=False):
        self.cfg = POLICIES[name]
        self.m = build(scn, step_h, self.cfg["friction"], arm)
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qa = np.array([m.jnt_qposadr[i] for i in jid])
        self.va = np.array([m.jnt_dofadr[i] for i in jid])
        self.act = np.array([next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == i) for i in jid])
        self.tau_lim = m.actuator_ctrlrange[self.act, 1]
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.pol = torch.jit.load(os.path.join(ROOT, "policies", "go2", self.cfg["file"]), map_location="cpu").eval()
        self.last_a = np.zeros(12)
        self.target = Q0.copy()
        d = self.d
        d.qpos[0:3], d.qpos[3:7] = [0, 0, 0.36], [1, 0, 0, 0]
        d.qpos[self.qa] = Q0
        if arm:
            aj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"z1_joint{i}") for i in range(1, 7)]
            d.qpos[[m.jnt_qposadr[i] for i in aj]] = Z1_STOW
            arm_act = [next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == i) for i in aj]
            d.ctrl[arm_act] = Z1_STOW  # Z1 position servos hold the folded pose (arm = static payload)
        mujoco.mj_forward(m, d)
        self.hist = None
        self.primed = False

    def single_obs(self, cmd):
        d, c = self.d, self.cfg
        R = d.xmat[self.base].reshape(3, 3)
        w = d.qvel[3:6] * 0.25
        g = R.T @ np.array([0, 0, -1.0])
        cs = cmd * np.array(c["cmd_scale"])
        q, dq = d.qpos[self.qa] - Q0, d.qvel[self.va] * 0.05
        if c["kind"] == "himloco":
            return np.concatenate([cs, w, g, q, dq, self.last_a])
        return np.concatenate([w, g, cs, q, dq, self.last_a])

    def act_policy(self, cmd):
        o = torch.tensor(self.single_obs(cmd), dtype=torch.float32)[None]
        with torch.no_grad():
            if self.cfg["kind"] == "internal_hist":
                if not self.primed:  # Isaac Lab fills the history with the first observation after reset
                    for _ in range(9):
                        self.pol(o)
                    self.primed = True
                a = self.pol(o)[0].numpy()
            else:  # HIMLoco: newest frame first
                if self.hist is None:
                    self.hist = [o] * self.cfg["hist"]
                self.hist = [o] + self.hist[:-1]
                a = self.pol(torch.cat(self.hist, -1))[0].numpy()
        a = np.clip(a, -100, 100)
        self.last_a = a.astype(float)
        self.target = Q0 + self.cfg["act_scale"] * self.last_a

    def step(self, cmd):
        self.act_policy(cmd)
        d, c = self.d, self.cfg
        for _ in range(DECIM):
            tau = c["kp"] * (self.target - d.qpos[self.qa]) - c["kd"] * d.qvel[self.va]
            d.ctrl[self.act] = np.clip(tau, -self.tau_lim, self.tau_lim)
            mujoco.mj_step(self.m, d)


def run(name, scn, step_h, T, video=None, arm=False):
    r = Runner(name, scn, step_h, arm)
    n = int(T / (DT * DECIM))
    frames, log = [], dict(t=[], pos=[], v=[], w=[], cmd=[], up=[], g=[])
    if video:
        ren = mujoco.Renderer(r.m, 360, 640)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, r.base
        cam.distance, cam.azimuth, cam.elevation = 1.6, 120, -15
    fell_at = None
    for _ in range(n):
        t = r.d.time
        cmd = command(scn, t)
        r.step(cmd)
        R = r.d.xmat[r.base].reshape(3, 3)
        log["t"].append(r.d.time); log["pos"].append(r.d.xpos[r.base].copy()); log["cmd"].append(cmd)
        log["v"].append(R.T @ r.d.qvel[0:3]); log["w"].append(r.d.qvel[3:6].copy()); log["up"].append(R[2, 2]); log["g"].append(R.T @ np.array([0, 0, -1.0]))
        if fell_at is None and R[2, 2] < 0.5:
            fell_at = r.d.time
        if video:
            ren.update_scene(r.d, camera=cam)
            frames.append(label(ren.render(), f"{r.cfg['label']}{' + Z1 arm' if arm else ''}  |  {scn}  t={r.d.time:4.1f}s  cmd=({cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f})"))
        if fell_at is not None and r.d.time - fell_at > 1.5:
            break
    L = {k: np.array(v) for k, v in log.items()}
    moving = np.linalg.norm(L["cmd"], axis=1) > 0
    ok = L["t"] < (fell_at or 1e9)
    sel = moving & ok
    rp = np.degrees(np.arctan2(L["g"][ok, :2] * [1, -1], -L["g"][ok, 2:3]))  # roll, pitch
    res = dict(policy=name, arm=arm, scenario=scn, step_height=step_h if scn == "stairs" else None, fell=fell_at is not None,
               survived_s=float(fell_at or L["t"][-1]),
               lin_vel_err=float(np.mean(np.linalg.norm(L["cmd"][sel, :2] - L["v"][sel, :2], axis=1))) if sel.any() else None,
               yaw_err=float(np.mean(np.abs(L["cmd"][sel, 2] - L["w"][sel, 2]))) if sel.any() else None,
               x_travel=float(L["pos"][ok, 0].max()), max_base_z=float(L["pos"][ok, 2].max()),
               mean_base_z=float(L["pos"][ok, 2].mean()), max_abs_roll_deg=float(np.abs(rp[:, 1]).max()),
               max_abs_pitch_deg=float(np.abs(rp[:, 0]).max()))
    if scn == "stairs":
        end_x = 1.0 + 0.3 * 6 * 2 + 1.2
        res["climbed_up"] = bool(res["max_base_z"] > 0.25 + step_h * 6)
        res["completed"] = bool(L["pos"][ok, 0].max() > end_x + 0.3 and not res["fell"])
    return res, frames


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, img.shape[1], 18], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((5, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="all", help="all, or comma list of: " + ",".join(POLICIES))
    ap.add_argument("--scenario", choices=["walk", "turn", "stairs", "all"], default="all")
    ap.add_argument("--arm", action="store_true", help="mount the Unitree Z1 (folded, static payload)")
    ap.add_argument("--step-height", type=float, default=0.10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", "locomotion")
    os.makedirs(out, exist_ok=True)
    import imageio
    pols = list(POLICIES) if args.policy == "all" else args.policy.split(",")
    scns = ["walk", "turn", "stairs"] if args.scenario == "all" else [args.scenario]
    tag = ("_arm" if args.arm else "") + args.tag
    rows = []
    for scn in scns:
        T = {"walk": 16.0, "turn": 17.0, "stairs": 20.0}[scn]
        vids = []
        for p in pols:
            res, frames = run(p, scn, args.step_height, T, video=True, arm=args.arm)
            print(json.dumps(res)); rows.append(res)
            imageio.mimsave(os.path.join(out, f"{scn}__{p}{tag}.mp4"), frames, fps=50, quality=7)
            vids.append(frames)
        if len(vids) > 1:  # side-by-side comparison
            n = max(len(v) for v in vids)
            pad = lambda v: v + [v[-1]] * (n - len(v))
            vids = [pad(v) for v in vids]
            imageio.mimsave(os.path.join(out, f"{scn}__comparison{tag}.mp4"),
                            [np.concatenate([v[i] for v in vids], 1) for i in range(n)], fps=50, quality=7)
    with open(os.path.join(out, f"summary{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
