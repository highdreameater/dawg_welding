# Live demo command sheet

## 0. Before the talk (once)

```bash
cd "/Users/highdreameater/Desktop/main/mani/Project "   # the folder name ends in a space, so keep the quotes
conda activate spot_weld
```

- On macOS, live 3-D windows (`--viewer`) need **`mjpython`**, not `python`.
- Close the viewer window to end a run.
- Do a dry run of each live command before the talk so the first launch (model compile) isn't on stage.

---

## Live: runs in seconds, opens a 3-D window

| Slide | What it shows | Command |
|---|---|---|
| 6 WBC | Go2 + Z1 holds still while the tool traces a seam (final tuned controller) | `mjpython sim/stance_demo.py --robot go2_z1_official --scenario weld --viewer` |
| 6 / 11 | Fast large arm sweeps: the body stays put | `mjpython sim/stance_demo.py --robot go2_z1_official --scenario stress --viewer` |
| 7 force | 5 N force control over a curved pipe, ±30° | `mjpython sim/force_demo.py --viewer` |
| 7 (robustness) | The real pipe is moved, rotated and resized vs the map; vision still finds it | `mjpython sim/force_demo.py --pipe-dx 0.04 --pipe-dy 0.06 --pipe-dyaw 6 --pipe-radius 0.12 --viewer` |
| 7 (contrast) | Trust the map instead of the camera, using the same moved pipe | `mjpython sim/force_demo.py --perception map --pipe-dx 0.04 --pipe-dy 0.06 --pipe-dyaw 6 --pipe-radius 0.12 --viewer` |

## Live: numbers in the terminal, no window (~3–5 s each)

| Slide | Prints | Command |
|---|---|---|
| 6 / 11 | body drift 0.40 mm, tool RMS 0.08 mm, QP 0.19 ms | `python sim/stance_demo.py --robot go2_z1_official --scenario weld --no-video --tag _live` |
| 11 | the earlier controller's untuned settings (~4 mm tool error), for contrast | `python sim/stance_demo.py --robot go2_z1 --scenario weld --no-video --tag _live` |
| 7 | mean force, RMS error, contact lost | `python sim/force_demo.py --no-video --tag _live` |
| 11 ablation | metrics + plots for the variants | `python sim/wbc_tuning_study.py --no-video` |

**Use `go2_z1_official` for the slide numbers.** Plain `go2_z1` gives 4.1 mm tool error, not 0.08 mm.

---

## Too slow to run live: play the recordings

| Slide | Command |
|---|---|
| 5 / 2 walking + stairs with the arm | `open results/locomotion/stairs__moe_cts_arm.mp4` |
| 5 pretrained vs fine-tuned | `open results/locomotion/walk__comparison_arm.mp4` |
| 6 / 11 WBC | `open results/stance_wbc/go2_z1/stress.mp4` |
| 7 force on the pipe | `open results/force_control/go2_z1_vision_model.mp4` |
| 8 surface survey + trace | `open results/surface/go2_z1_seed4_wrist.mp4` |
| 9 / 12 full mission (seed 3) | `open results/curved_mission/mission_seed3_video.mp4` |
| 9 exploration map | `open results/curved_mission/explore_seed3.png` |
| 12 force maps | `open results/curved_mission/paint_seed3.png` |

---

## Full mission LIVE (slides 4, 9, 12): start it, Ctrl-C when you've shown enough

Seed 3 is the run on the slides: same noise, so the same order **5 → 23 → 7 → 37** and the same numbers (deterministic).

```bash
# 3-D window + log in the terminal (camera follows the robot; scroll = zoom, drag = rotate)
mjpython sim/curved_mission.py --viewer --no-video --seed 3 --tag _live

# the other strategy: explore everything first, then work in the cheapest-tour order (seed 1 went 37 → 7 → 23 → 5)
mjpython sim/curved_mission.py --viewer --no-video --seed 3 --strategy map_first --tag _live_mapfirst

# no window, just the decision log scrolling (narrate the layers from it)
python sim/curved_mission.py --no-video --seed 3 --tag _live
```

- **Stop:** press **Ctrl-C** in the terminal, or close the 3-D window. It prints `STOPPED BY USER` and still saves the partial results (`results/curved_mission/*_live*`).
- **Full length:** ~705 s robot time. Headless that's ~4–5 min on this laptop; the window makes it slower.
- **What to point at in the log** (each line is one layer of slide 4):
  - `[decide] explore -> viewpoint ... (unseen surface N, d m)`: the explore score from slide 9.
  - `look around: ... localisation error 0 cm / 0.2 deg`: map matching working.
  - `[2 survey] marker 7: surface model 1395 cells ... vs real surface 0.2 mm / 0.9 deg RMS`: slide 8.
  - `[3 plan reach] ... standoff 0.73 m, reachable 400 mm wide -> 1 patch(es)`: the reach plan.
  - `[4 dock] ... 7 mm / 1.0 deg from the marker-relative dock`: docking.
  - `area 7.0 done: F 5.01 N ... contact lost 0.0 %, base drift 0.2 mm`: slides 6–7.
- **Seed 3 timing, to skip ahead in your talk:** marker 5 done at ~150 s, 23 at ~255 s, 7 at ~556 s, 37 at ~703 s (robot time, shown in the log as `[ 555.9s]`).

The recorded run's numbers, if asked: `cat results/curved_mission/summary_seed3.json`

---

## If something goes wrong

- **`mjpython: command not found`:** run `conda activate spot_weld` first.
- **Window doesn't open:** fall back to the `open results/...mp4` line for that slide.
- **Clean up after the talk:** `find results -name "*_live*" -delete` removes the files the `--tag _live` runs wrote.
