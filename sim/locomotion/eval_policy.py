"""Evaluate walking policies with the arm on board (deployment code path: sim/locomotion/policy.py).

    python sim/locomotion/eval_policy.py                         # go2 (pretrained) vs go2_payload (fine-tuned)
    python sim/locomotion/eval_policy.py --locos go2 go2_payload --trials 20 --payload 1.0 1.25

1. Fixed command schedule (stand, forward 0.8, yaw 0.8, sideways 0.5, back 0.5): velocity tracking, tilt, height.
2. Random trials: 20 s each, a new random command every 4 s (vx, vy, wz uniform in +-cmd_max: default the
   full training range +-1; --cmd-max 0.6 0.35 0.8 is the envelope the mission's path follower uses); counts falls.
   Repeated for each payload scale (arm + torch mass x scale).
Output: results/locomotion/eval.md, eval.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locomotion.policy import SIM_DT, VelocityPolicy  # noqa: E402
from wbc.robots import ROBOTS, ROOT, build_model  # noqa: E402
from wbc.stance_wbc import Indexer, initial_state  # noqa: E402

SCHEDULE = [(2, (0, 0, 0)), (6, (0.8, 0, 0)), (9, (0, 0, 0.8)), (12, (0, 0.5, 0)), (15, (-0.5, 0, 0)), (17, (0, 0, 0))]
WINDOWS = [((1, 2), "stand"), ((3, 6), "fwd 0.8"), ((7, 9), "yaw 0.8"), ((10, 12), "side 0.5"), ((13, 15), "back 0.5")]


def make(robot, payload):
    cfg = ROBOTS[robot]()
    m = build_model(cfg, workpiece=False, render_size=(64, 64))
    m.opt.timestep = SIM_DT
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    for b, nm in enumerate(names):
        if nm.startswith(cfg.arm_prefix) or nm == "torch":
            m.body_mass[b] *= payload
            m.body_inertia[b] *= payload
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    return cfg, m, d, Indexer(m, cfg)


def rollout(m, d, ix, pol, cmd_fn, T):
    pol.reset()
    log = []
    for _ in range(int(T / SIM_DT)):
        c = cmd_fn(d.time)
        pol.torques(m, d, c)
        mujoco.mj_step(m, d)
        R = d.xmat[ix.base].reshape(3, 3)
        log.append((d.time, *c, *(R.T @ d.qvel[:3]), d.qvel[5], d.qpos[2], np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        if d.qpos[2] < 0.12 or R[2, 2] < 0.5:
            return np.array(log), d.time
    return np.array(log), None


def schedule_cmd(t):
    for te, c in SCHEDULE:
        if t < te:
            return np.array(c, float)
    return np.zeros(3)


def evaluate(robot, loco, payloads, trials, cmd_max=(1.0, 1.0, 1.0), seed=0):
    out = dict(loco=loco)
    cfg, m, d, ix = make(robot, 1.0)
    pol = VelocityPolicy(m, cfg, loco=loco)
    initial_state(m, d, ix)
    L, fell = rollout(m, d, ix, pol, schedule_cmd, 17.0)
    sched = {}
    for (ta, tb), name in WINDOWS:
        s = (L[:, 0] > ta) & (L[:, 0] < tb)
        if s.any():
            sched[name] = dict(vx=float(L[s, 4].mean()), vy=float(L[s, 5].mean()), wz=float(L[s, 7].mean()),
                               z=float(L[s, 8].mean()), tilt_max=float(L[s, 9].max()))
    out["schedule"] = dict(fell_at=fell, **sched)
    rng = np.random.default_rng(seed)
    out["random"] = {}
    for p in payloads:
        cfg, m, d, ix = make(robot, p)
        pol = VelocityPolicy(m, cfg, loco=loco)
        falls, err, tilt = 0, [], []
        for _ in range(trials):
            cmds = rng.uniform(-1, 1, (5, 3)) * np.asarray(cmd_max)
            initial_state(m, d, ix, yaw=rng.uniform(-np.pi, np.pi))
            L, fell = rollout(m, d, ix, pol, lambda t: cmds[min(int(t // 4), 4)], 20.0)
            falls += fell is not None
            settled = (L[:, 0] % 4) > 1.0  # 1 s after each command change
            err.append(np.sqrt(np.mean(np.sum((L[settled, 1:3] - L[settled, 4:6]) ** 2, 1))))
            tilt.append(L[:, 9].max())
        out["random"][f"payload x{p:g}"] = dict(falls=falls, trials=trials, lin_vel_rms_err=float(np.mean(err)),
                                                 tilt_max=float(np.max(tilt)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="go2_z1")
    ap.add_argument("--locos", nargs="+", default=["go2", "go2_payload"])
    ap.add_argument("--payload", nargs="+", type=float, default=[1.0, 1.25])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--cmd-max", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="random command range vx vy wz")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    res = [evaluate(args.robot, lo, args.payload, args.trials, args.cmd_max) for lo in args.locos]
    out = os.path.join(ROOT, "results", "locomotion")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"eval{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    md = ["| policy | " + " | ".join(n for _, n in WINDOWS[1:]) + " | base height (fwd) | max tilt | "
          + " | ".join(f"falls / RMS vel err, payload x{p:g}" for p in args.payload) + " |",
          "|---" * (3 + len(WINDOWS) - 1 + len(args.payload)) + "|"]
    for r in res:
        s = r["schedule"]
        cells = [f"{s['fwd 0.8']['vx']:.2f}", f"{s['yaw 0.8']['wz']:.2f}", f"{s['side 0.5']['vy']:.2f}",
                 f"{s['back 0.5']['vx']:.2f}"] if s["fell_at"] is None else ["fell"] * 4
        tilt = max(v["tilt_max"] for k, v in s.items() if isinstance(v, dict))
        z = s["fwd 0.8"]["z"] if "fwd 0.8" in s else float("nan")
        rnd = [f"{v['falls']}/{v['trials']}, {v['lin_vel_rms_err']:.2f} m/s" for v in r["random"].values()]
        md.append(f"| {r['loco']} | " + " | ".join(cells) + f" | {z:.2f} m | {tilt:.0f}° | " + " | ".join(rnd) + " |")
    txt = "\n".join(md)
    with open(os.path.join(out, f"eval{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(f"robot={args.robot}; schedule = achieved velocity for the commanded one\n\n{txt}\n")
    print(txt)


if __name__ == "__main__":
    main()
