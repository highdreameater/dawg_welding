"""Stance WBC tuning study: the tuned controller next to the earlier / untuned settings it was designed away from.

Every variant runs the same scenario as sim/stance_demo.py (same plant, plan, metrics) with ONE design choice undone
(or one model error added), so each video shows what that choice buys.

  weld (torch-tip precision, base held):
    tuned                final controller (go2_z1_official wbc_gains)
    v0_defaults          first version: StanceWBC defaults (tool kp 900 / kd 60, no integral, no disturbance observer)
    no_friction_comp     tuned gains, but the Z1's 1 Nm joint friction is not compensated (no feed-forward, no observer)
    friction_ff_only     fixed Coulomb feed-forward instead of the arm disturbance observer
    no_armature          Menagerie Z1 without gearbox rotor inertia (light wrist + 1 Nm friction -> torque chatter)
    rate_100hz/50hz/25hz QP solved at 100 / 50 / 25 Hz instead of 500 Hz (torques held in between)
    wrong_tool_mass      plant carries a 1.2 kg torch, controller believes there is none
  stress (fast large arm sweeps, base held):
    soft_base            base task kp 100 / kd 20 instead of 900 / 60
    very_soft_base       base task kp 25 / kd 10
    underdamped_base     base task kd 6 instead of 60 (kp 900)
    very_underdamped     base task kd 1: the base rings, feet slip
    tool_3kg_unmodelled  plant carries a 3 kg tool the controller does not know about
  metrics only (made no difference - also a result):
    weak_base_weight     base task weight 1 instead of 100 (tasks are compatible, so weights barely matter)
    no_contact_stab      kc = 0: no velocity feedback on the stance-foot constraint
    too_stiff_base       base task kp 20000 / kd 300 (holds tighter, but more jitter and torque)

    python sim/wbc_tuning_study.py                         # everything -> results/stance_wbc/tuning/
    python sim/wbc_tuning_study.py --variants tuned soft_base --scenario stress
    python sim/wbc_tuning_study.py --no-video              # metrics + plots only (fast)
"""
import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stance_demo import DT, plot, rot_dev_deg, run  # noqa: E402
from wbc.robots import ROBOTS, ROOT  # noqa: E402

ROBOT = "go2_z1_official"
OUT = os.path.join(ROOT, "results", "stance_wbc", "tuning")


def _no_friction(w):
    w.friction[:] = 0.0            # no Coulomb feed-forward


def _rate(n):
    """QP solved every n-th physics tick (500 / n Hz), torques held in between (zero-order hold)."""
    def hook(w):
        f, st = w.compute, dict(k=-1)

        def compute(qpos, qvel, ref):
            st["k"] += 1
            if st["k"] % n == 0:
                st["out"] = f(qpos, qvel, ref)
            return st["out"]
        w.compute = compute
    return hook


VARIANTS = {
    "tuned": dict(),
    "v0_defaults": dict(wbc=dict(kp_ee=900.0, kd_ee=60.0, ki_ee=0.0, dob_T=None, dob_cartesian_only=False)),
    "no_friction_comp": dict(wbc=dict(dob_T=None), hook=_no_friction),
    "friction_ff_only": dict(wbc=dict(dob_T=None)),
    "no_armature": dict(cfg=dict(arm_armature=None)),
    "rate_100hz": dict(hook=_rate(5)),
    "rate_50hz": dict(hook=_rate(10)),
    "rate_25hz": dict(hook=_rate(20)),
    "wrong_tool_mass": dict(torch=(1.2, 0.0)),
    "soft_base": dict(wbc=dict(kp_base=100.0, kd_base=20.0)),
    "very_soft_base": dict(wbc=dict(kp_base=25.0, kd_base=10.0)),
    "underdamped_base": dict(wbc=dict(kd_base=6.0)),
    "very_underdamped": dict(wbc=dict(kp_base=900.0, kd_base=1.0)),
    "tool_3kg_unmodelled": dict(torch=(3.0, 0.0)),
    "weak_base_weight": dict(wbc=dict(w_base=1.0)),
    "no_contact_stab": dict(wbc=dict(kc=0.0)),
    "too_stiff_base": dict(wbc=dict(kp_base=20000.0, kd_base=300.0)),
}
LABELS = {
    "tuned": "TUNED (final)", "v0_defaults": "first version (default gains)", "no_friction_comp": "no friction compensation",
    "friction_ff_only": "friction FF, no observer", "no_armature": "no Z1 rotor inertia", "rate_100hz": "QP at 100 Hz",
    "rate_50hz": "QP at 50 Hz", "rate_25hz": "QP at 25 Hz", "wrong_tool_mass": "unmodelled 1.2 kg torch",
    "soft_base": "soft base (kp 100)", "very_soft_base": "very soft base (kp 25)", "underdamped_base": "base kd 6",
    "very_underdamped": "base kd 1 (underdamped)", "tool_3kg_unmodelled": "unmodelled 3 kg tool",
}
# which variants get a video per scenario (the rest: metrics + plots); the first 6 also go into the grid video
VIDEO = {
    "weld": ["tuned", "v0_defaults", "no_friction_comp", "rate_100hz", "rate_50hz", "rate_25hz",
             "friction_ff_only", "no_armature", "wrong_tool_mass"],
    "stress": ["tuned", "soft_base", "very_soft_base", "very_underdamped", "tool_3kg_unmodelled", "no_friction_comp",
               "underdamped_base"],
}


def one(job):
    name, scn, render = job
    v = VARIANTS[name]
    cfg = ROBOTS[ROBOT]()
    if "cfg" in v:
        cfg = dataclasses.replace(cfg, **v["cfg"])
    tm, ctm = v.get("torch", (0.0, 0.0))
    try:
        res, L, frames, plan = run(ROBOT, scn, tm, ctm, render=render, cfg=cfg, wbc_kw=v.get("wbc"),
                                   wbc_hook=v.get("hook"), label=name)
    except Exception as ex:                                     # a variant may blow up: report, keep going
        return dict(variant=name, scenario=scn, error=repr(ex))
    res["variant"] = name
    plot(L, plan, name, scn, os.path.join(OUT, "plots", f"{scn}__{name}.png"))
    if frames:
        import imageio
        imageio.mimsave(os.path.join(OUT, f"{scn}__{name}.mp4"), frames, fps=50, quality=7)
    i0 = int(plan.arm_start / DT)
    np.savez_compressed(os.path.join(OUT, "data", f"{scn}__{name}.npz"), t=L["t"],
                        drift_mm=1000 * np.linalg.norm(L["base_pos"] - L["base_pos"][i0], axis=1),
                        tilt_deg=rot_dev_deg(L["base_quat"][i0], L["base_quat"]),
                        tip_err_mm=1000 * np.linalg.norm(L["tip"] - L["tip_ref"], axis=1),
                        phases=np.array([t for t, _ in plan.phases]), phase_names=np.array([n for _, n in plan.phases]))
    return res


def table(rows):
    cols = ["variant", "scenario", "fell", "base_pos_dev_max_mm", "base_rot_dev_max_deg", "base_vel_rms_mm_s",
            "torch_err_rms_mm", "torch_err_max_mm", "foot_slip_max_mm", "peak_torque_frac", "qp_failures"]
    fmt = lambda x: "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows if "error" not in r]
    md += [f"| {r['variant']} | {r['scenario']} | error: {r['error']} |" for r in rows if "error" in r]
    return "\n".join(md)


def compare_plot(scn, names):
    """Base drift, base tilt and (weld) tool error of all variants on one figure, log scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("drift_mm", "base drift [mm]", 1e-2), ("tilt_deg", "base tilt [deg]", 1e-3)]
    if scn == "weld":
        keys.append(("tip_err_mm", "torch tip error [mm]", 1e-2))
    fig, ax = plt.subplots(len(keys), 1, figsize=(11, 2.8 * len(keys)), sharex=True)
    for nm in names:
        p = os.path.join(OUT, "data", f"{scn}__{nm}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for a, (k, _, _) in zip(ax, keys):
            y = np.where(z[k] > 0, z[k], np.nan)
            a.plot(z["t"], y, lw=2.2 if nm == "tuned" else 1.0, color="k" if nm == "tuned" else None, label=nm,
                   zorder=10 if nm == "tuned" else 1)
    for a, (_, lab, lo) in zip(ax, keys):
        a.set_yscale("log"); a.set_ylabel(lab); a.grid(alpha=0.3, which="both")
        a.set_ylim(bottom=lo)
        for t0, n in zip(z["phases"], z["phase_names"]):
            a.axvline(t0, color="0.8", lw=0.8)
    for t0, n in zip(z["phases"], z["phase_names"]):
        ax[0].text(t0 + 0.05, 0.92, n, fontsize=7, color="0.4", transform=ax[0].get_xaxis_transform())
    ax[0].legend(ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08))
    ax[-1].set_xlabel("time [s]")
    fig.suptitle(f"Stance WBC tuning study - {scn} (tuned = black)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{scn}__comparison.png"), dpi=120)
    plt.close(fig)


def grid_video(scn, names, cols=3):
    """Side-by-side video of up to 6 variants (same time base), each tile labelled with its live overlay."""
    import imageio
    import cv2
    names = [n for n in names if os.path.exists(os.path.join(OUT, f"{scn}__{n}.mp4"))][:2 * cols]
    if len(names) < 2:
        return
    clips = [imageio.mimread(os.path.join(OUT, f"{scn}__{n}.mp4"), memtest=False) for n in names]
    h, w = 360, 640
    n = min(len(c) for c in clips)
    rows = (len(names) + cols - 1) // cols
    out = []
    for k in range(n):
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, c in enumerate(clips):
            r, cc = divmod(i, cols)
            tile = np.ascontiguousarray(cv2.resize(c[k][..., :3], (w, h), interpolation=cv2.INTER_AREA))
            cv2.rectangle(tile, (0, h - 46), (w, h), (0, 0, 0), -1)
            cv2.putText(tile, f"{i + 1}. {LABELS.get(names[i], names[i])}", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (120, 255, 120) if names[i] == "tuned" else (255, 255, 255), 2, cv2.LINE_AA)
            canvas[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = tile
        out.append(canvas)
    imageio.mimsave(os.path.join(OUT, f"{scn}__grid.mp4"), out, fps=50, quality=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--scenario", choices=["weld", "stress", "all"], default="all")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--compose-only", action="store_true", help="rebuild comparison plots + grid videos from saved runs")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    for sub in ("", "plots", "data"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    scns = ["weld", "stress"] if args.scenario == "all" else [args.scenario]
    for s in scns:
        if args.compose_only:
            compare_plot(s, VIDEO[s])
            grid_video(s, VIDEO[s])
    if args.compose_only:
        return
    jobs = [(v, s, not args.no_video and v in VIDEO[s]) for s in scns for v in args.variants]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    for s in scns:
        compare_plot(s, VIDEO[s])
        if not args.no_video:
            grid_video(s, VIDEO[s])
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    md = table(rows)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(f"Stance WBC tuning study, robot={ROBOT}\n\n{md}\n")
    print(md)


if __name__ == "__main__":
    main()
