"""Force control demo: the robot stands at the pipe station and presses the torch tip on the curved pipe surface
with a constant normal force while sweeping an arc over the pipe top (see sim/force_task.py).

    python sim/force_demo.py                              # 5 N, +-30 deg arc, 2 passes -> results/force_control/
    python sim/force_demo.py --force 10 --passes 4
    python sim/force_demo.py --ctrl-torch-mass 0.6        # controller's torch-mass belief is wrong (plant 1.2 kg)
    python sim/force_demo.py --force-noise 0.3            # Gaussian noise on the force measurement [N]
    python sim/force_demo.py --viewer                     # live window (macOS needs mjpython)
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from force_task import PipeForceTask, tip_contact_force  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model, controller_model  # noqa: E402
from wbc.stance_wbc import Indexer, StanceWBC, initial_state  # noqa: E402
from world import WORLDS  # noqa: E402

DT = 0.002


class ForceSensor:
    """Contact-tip force 'sensor': MuJoCo contact force + optional Gaussian noise, first-order low-pass filtered."""

    def __init__(self, m, noise=0.0, tau=0.005, seed=0):
        self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "contact_tip")
        self.noise, self.alpha = noise, DT / (tau + DT)
        self.rng = np.random.default_rng(seed)
        self.f = 0.0
        self.raw = 0.0

    def read(self, m, d):
        self.raw, _ = tip_contact_force(m, d, self.gid)
        z = self.raw + (self.rng.normal(0, self.noise) if self.noise > 0 else 0.0)
        self.f += self.alpha * (z - self.f)
        return self.f


class Recorder:
    def __init__(self, plant, cfg, cam):
        self.renderer = mujoco.Renderer(plant, 480, 854)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = cam["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        self.frames = []

    def frame(self, d, text, tip=None, n=None, force=0.0):
        self.renderer.update_scene(d, camera=self.cam)
        scn = self.renderer.scene
        if tip is not None and force > 0.05 and scn.ngeom < scn.maxgeom:
            # contact-force arrow (2 cm per N) pointing into the surface
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.1, 0.8, 1.0, 1], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, tip + (0.02 * force + 0.01) * n, tip + 0.01 * n)
            scn.ngeom += 1
        from PIL import Image, ImageDraw
        im = Image.fromarray(self.renderer.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width, 38], fill=(0, 0, 0))
        dr.multiline_text((6, 3), text, fill=(255, 255, 255))
        self.frames.append(np.asarray(im))


def run(args, cfg, world, render=True, viewer=False):
    st = world.stations["pipe"]
    plant = build_model(cfg, torch_mass=args.torch_mass, world=world, contact_tip=True)
    ctrl = controller_model(cfg, args.ctrl_torch_mass)
    plant.opt.timestep = ctrl.opt.timestep = DT
    d = mujoco.MjData(plant)
    ix = Indexer(plant, cfg)
    initial_state(plant, d, ix, xy=st.base_xy, yaw=st.base_yaw)
    task = PipeForceTask(cfg, ctrl, d.qpos.copy(), st.work, f_des=args.force, theta_max=np.radians(args.arc_deg),
                         passes=args.passes, speed=args.speed)
    wbc = StanceWBC(ctrl, cfg, **cfg.wbc_gains)
    sensor = ForceSensor(plant, noise=args.force_noise)
    rec = Recorder(plant, cfg, dict(lookat=[task.x_work - 0.25, st.work.y, 0.45], distance=1.9, azimuth=200,
                                    elevation=-22)) if render else None
    v = None
    if viewer:
        from mujoco import viewer as mj_viewer
        v = mj_viewer.launch_passive(plant, d)

    L = {k: [] for k in ["t", "f", "f_raw", "f_des", "phase", "tip", "tip_ref", "n", "axis", "base_pos", "base_quat",
                         "feet", "tau", "vn_des"]}
    solve_ms = []
    for k in range(int(task.T / DT)):
        t = d.time
        f_meas = sensor.read(plant, d)
        ref = task.ref(t, d.site_xpos[ix.tip].copy(), d.qpos[ix.arm_q].copy(), f_meas)
        t0 = time.perf_counter()
        tau, _ = wbc.compute(d.qpos.copy(), d.qvel.copy(), ref)
        solve_ms.append(1000 * (time.perf_counter() - t0))
        d.ctrl[:] = tau[ix.act_dof]
        mujoco.mj_step(plant, d)
        ph = task.phase(t)
        n = ref["force"]["n"] if "force" in ref else task.pipe.surface_point(task.x_work, task.theta(t))[1]
        L["t"].append(t); L["f"].append(f_meas); L["f_raw"].append(sensor.raw); L["phase"].append(ph)
        L["f_des"].append(ref["force"]["f_des"] if "force" in ref else np.nan)
        L["tip"].append(d.site_xpos[ix.tip].copy()); L["tip_ref"].append(ref.get("ee_pos", np.full(3, np.nan)))
        L["n"].append(n); L["axis"].append(d.site_xmat[ix.tip].reshape(3, 3)[:, 0].copy())
        L["base_pos"].append(d.qpos[0:3].copy()); L["base_quat"].append(d.qpos[3:7].copy())
        L["feet"].append(d.geom_xpos[ix.feet].copy()); L["tau"].append(d.actuator_force.copy())
        L["vn_des"].append(wbc.vn_des if "force" in ref else np.nan)
        if v is not None:
            if not v.is_running():
                break
            v.sync()
            time.sleep(max(0.0, DT - 1e-4))
        if rec is not None and k % 10 == 9:
            base_dev = 1000 * np.linalg.norm(d.qpos[0:3] - L["base_pos"][int(task.start["unstow"] / DT)]) if t > task.start["unstow"] else 0.0
            txt = (f"force control  {cfg.name}  t={t:5.2f}s  [{ph}]\n"
                   f"F = {f_meas:5.2f} N (target {args.force:.1f})   arc {np.degrees(task.theta(t)):+5.1f} deg   "
                   f"base drift {base_dev:5.2f} mm")
            rec.frame(d, txt, d.site_xpos[ix.tip].copy(), n, f_meas)
    if v is not None:
        v.close()
    L = {k: (np.array(x) if k != "phase" else np.array(x, dtype=object)) for k, x in L.items()}
    return L, task, wbc, plant, solve_ms, (rec.frames if rec else [])


def rot_dev_deg(q_ref, Q):
    out = []
    for q in Q:
        e = np.zeros(3); mujoco.mju_subQuat(e, q, q_ref); out.append(np.degrees(np.linalg.norm(e)))
    return np.array(out)


def metrics(L, task, wbc, plant, solve_ms, args):
    t, ph = L["t"], L["phase"]
    sweep = ph == "sweep"
    contact = np.isin(ph, ["sweep", "dwell"])
    i0 = int(task.start["unstow"] / DT)
    after = t >= task.start["unstow"]
    f, fe = L["f_raw"], L["f_raw"] - args.force
    tip_e = L["tip"] - L["tip_ref"]
    n = L["n"]
    tan_e = 1000 * np.linalg.norm(tip_e - np.sum(tip_e * n, 1, keepdims=True) * n, axis=1)
    axis_e = np.degrees(np.arccos(np.clip(np.sum(L["axis"] * -n, 1), -1, 1)))
    app = ph == "approach"
    t_contact = t[app][np.argmax(f[app] > 0.05)] - task.start["approach"] if np.any(f[app] > 0.05) else np.nan
    pos_dev = 1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1)
    slip = 1000 * np.max(np.linalg.norm((L["feet"] - L["feet"][i0])[:, :, :2], axis=2), axis=1)
    return dict(
        robot=args.robot, f_des=args.force, arc_deg=args.arc_deg, passes=args.passes, speed_mm_s=1000 * args.speed,
        torch_mass=args.torch_mass, ctrl_torch_mass=args.ctrl_torch_mass, force_noise=args.force_noise,
        sweep_f_mean=float(f[sweep].mean()), sweep_f_rms_err=float(np.sqrt(np.mean(fe[sweep] ** 2))),
        sweep_f_min=float(f[sweep].min()), sweep_f_max=float(f[sweep].max()),
        contact_lost_frac=float(np.mean(f[contact] < 0.05)),
        approach_contact_s=float(t_contact), approach_f_peak=float(f[app].max()),
        tan_err_rms_mm=float(np.sqrt(np.mean(tan_e[sweep] ** 2))), tan_err_max_mm=float(tan_e[sweep].max()),
        axis_err_rms_deg=float(np.sqrt(np.mean(axis_e[sweep] ** 2))), axis_err_max_deg=float(axis_e[sweep].max()),
        base_pos_dev_max_mm=float(pos_dev[after].max()),
        base_rot_dev_max_deg=float(rot_dev_deg(L["base_quat"][i0], L["base_quat"])[after].max()),
        foot_slip_max_mm=float(slip[after].max()),
        peak_torque_frac=float(np.max(np.abs(L["tau"][after]) / plant.actuator_forcerange[:, 1])),
        arc_ik_residual_mm=float(task.arc_ik_residual_mm),
        qp_ms_mean=float(np.mean(solve_ms)), qp_ms_max=float(np.max(solve_ms)), qp_failures=wbc.fail,
        fell=bool(np.any(L["base_pos"][:, 2] < 0.2)))


def plot(L, task, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = L["t"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, L["f_raw"], color="#cf222e", lw=0.8, label="measured (contact)")
    ax[0].plot(t, L["f_des"], "k--", lw=1, label="target")
    ax[0].set_ylabel("normal force [N]"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(t, np.degrees([task.theta(x) for x in t]), color="#1f6feb")
    ax[1].set_ylabel("arc angle [deg]")
    tip_e = L["tip"] - L["tip_ref"]
    n = L["n"]
    tan_e = 1000 * np.linalg.norm(tip_e - np.sum(tip_e * n, 1, keepdims=True) * n, axis=1)
    m = np.isin(L["phase"], ["approach", "sweep", "dwell"])
    ax[2].plot(t[m], tan_e[m], ".", ms=1, color="#8250df")
    ax[2].set_ylabel("tangential tip err [mm]")
    i0 = int(task.start["unstow"] / DT)
    a3 = t >= task.start["unstow"]  # after the (intentional) 4 cm brace
    ax[3].plot(t[a3], 1000 * np.linalg.norm(L["base_pos"][a3] - L["base_pos"][i0], axis=1), color="#1f6feb")
    ax[3].set_ylabel("base drift since unstow [mm]"); ax[3].set_xlabel("time [s]")
    for a in ax:
        for t0, nm in task.phases:
            a.axvline(t0, color="0.8", lw=0.8)
        a.grid(alpha=0.3)
    for t0, nm in task.phases:
        ax[0].text(t0 + 0.05, 0.92, nm, fontsize=7, color="0.4", transform=ax[0].get_xaxis_transform())
    fig.suptitle(f"Force control on a curved pipe — target {args.force:g} N, arc ±{args.arc_deg:g}°")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="anymal_c_z1")
    ap.add_argument("--world", choices=list(WORLDS), default="workshop")
    ap.add_argument("--force", type=float, default=5.0, help="target normal force [N]")
    ap.add_argument("--arc-deg", type=float, default=30.0, help="sweep +- this angle over the pipe top [deg]")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--speed", type=float, default=0.03, help="sweep speed along the surface [m/s]")
    ap.add_argument("--torch-mass", type=float, default=1.2)
    ap.add_argument("--ctrl-torch-mass", type=float, default=None)
    ap.add_argument("--force-noise", type=float, default=0.0, help="force sensor noise std [N]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.ctrl_torch_mass = args.torch_mass if args.ctrl_torch_mass is None else args.ctrl_torch_mass
    cfg, world = ROBOTS[args.robot](), WORLDS[args.world]()
    out = os.path.join(ROOT, "results", "force_control")
    os.makedirs(out, exist_ok=True)
    L, task, wbc, plant, solve_ms, frames = run(args, cfg, world, render=not (args.no_video or args.viewer), viewer=args.viewer)
    res = metrics(L, task, wbc, plant, solve_ms, args)
    print(json.dumps(res, indent=1))
    plot(L, task, args, os.path.join(out, f"pipe{args.tag}.png"))
    with open(os.path.join(out, f"summary{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    if frames:
        import imageio
        imageio.mimsave(os.path.join(out, f"pipe{args.tag}.mp4"), frames, fps=50, quality=7, macro_block_size=1)


if __name__ == "__main__":
    main()
