# dawg_welding: welding quadruped (Unitree Go2 / ANYmal-C + Unitree Z1) in MuJoCo

An arm-mounted quadruped that walks to a weld site, measures the workpiece, stands still and works on it with
force control. Simulation only.

```
A* path on the map -> RL locomotion (arm stowed = payload) -> depth-camera scan, cylinder fit -> dock on the measured pipe
   -> stance WBC holds the base rigid while the arm presses 5 N along a curved arc -> RL locomotion home
```

Two robots: **Unitree Go2 + Z1** (default) and ANYmal-C + Z1. Both share the stance WBC, the world, the planner,
the perception and the force task.

| component | status |
|---|---|
| Robot models: Go2 + Z1 and ANYmal-C + Z1, 1.2 kg welding torch | done (`sim/wbc/robots.py`) |
| Stance whole-body controller (QP) | done (`sim/wbc/stance_wbc.py`) |
| Environment: workshop with obstacles + two work stations (pipe, beam); the real pipe can differ from the map | done (`sim/world.py`) |
| Path planning on the known map (A* + smoothing, no SLAM) + path follower | done (`sim/planning/`) |
| Perception: head depth camera + cylinder fit (pipe position, direction, radius) | done (`sim/perception/`) |
| Force control on a curved surface (hybrid force/motion in the WBC), optional touch-guided tracing | done (`sim/force_task.py`, `sim/force_demo.py`) |
| Walking in MuJoCo: pretrained Isaac Lab Go2 / ANYmal-C rough policies, arm stowed as payload | done, sim-to-sim (`sim/locomotion/policy.py`) |
| Go2 walking policy fine-tuned with the arm payload (PPO in MuJoCo, payload curriculum) | done, trained on this laptop (`sim/locomotion/train_payload.py`, `policies/go2/go2_z1_payload/`) |
| Mission: plan → walk → scan → dock → force task → walk home | done (`sim/mission_demo.py`) |
| RL locomotion from scratch, ANYmal-C (ETH legged_gym, `anymal_c_rough`, Isaac Gym) | training scripts ready (`locomotion/`), to run on a Linux GPU server |

## Setup
```bash
conda create -n spot_weld python=3.11 && conda activate spot_weld
pip install -r requirements.txt
bash scripts/setup_assets.sh          # MuJoCo Menagerie: anybotics_anymal_c + unitree_go2 + unitree_z1 -> third_party/
```

## Stance WBC
```bash
python sim/stance_demo.py                                   # weld + stress -> results/stance_wbc/anymal_c_z1/
python sim/stance_demo.py --robot go2_z1                    # Go2 + Z1 (weld: torch error 0.19 mm RMS, base drift 0.5 mm)
python sim/stance_demo.py --scenario weld
python sim/stance_demo.py --ctrl-torch-mass 0.6             # controller believes the wrong torch mass (plant: 1.2 kg)
mjpython sim/stance_demo.py --scenario weld --viewer        # live window (macOS needs mjpython)
```

**Robot:** ANYmal-C (Menagerie) with a Unitree Z1 mounted on the base at (0.22, 0, 0.09) m and a 1.2 kg torch on the Z1 flange.
- Total mass 50.6 kg, of which arm + torch is 5.6 kg.
- Every joint is driven by a torque motor with the robot's limits: ANYmal 80 Nm, Z1 30 Nm (joint2 60 Nm).

**Controller** (`sim/wbc/stance_wbc.py`): an inverse-dynamics QP solved at 500 Hz with DAQP.
- Variables: joint accelerations and 3-D foot forces.
- Tasks: rigid base pose, 6-D torch-tip pose (PD plus clamped integral), arm posture.
- Hard constraints: floating-base dynamics, stance feet fixed, friction pyramids (μ=0.6), torque limits.
- Dynamics come from MuJoCo (`mj_fullM`, `qfrc_bias`, `mj_jac`, `mj_jacDot`), evaluated on the controller's own model copy. That copy can differ from the plant to test model error.
- Passive joint damping is included, and joint Coulomb friction is fed forward. The Z1 has 1 Nm per joint; without compensation it caused 3.6 mm of torch error.

**Weld sequence:** brace (lower base 4 cm), unstow arm, descend, weld a 30 cm seam at ~4 cm/s with ±4 mm weave, retract.
The Z1 reaches about 0.7 m, so the seam sits ~0.7 m ahead of the base centre with the torch pointing down. The workpiece is a narrow beam just ahead of the front feet.

| scenario | base drift | base tilt | base jitter | torch error RMS / max | peak torque | QP / tick |
|---|---|---|---|---|---|---|
| weld | 0.40 mm | 0.041° | 0.19 mm/s | 0.17 / 0.61 mm | 68% | 0.21 ms |
| stress (fast large arm sweeps) | 0.39 mm | 0.022° | 0.20 mm/s | – | 45% | 0.18 ms |
| weld, controller torch mass 0.6 kg vs real 1.2 kg | 0.41 mm | 0.125° | 0.20 mm/s | 0.32 / 1.09 mm | 55% | 0.20 ms |

All runs had no falls and no QP failures. Videos, plots and tables are in `results/stance_wbc/anymal_c_z1/`.

## Environment (`sim/world.py`)
A 10.5 m × 8 m workshop. The robot starts at (0, 0) facing +x. The same obstacle list builds the MuJoCo scene and the planner's map.
- **Obstacles:** walls, crate stacks, a steel column, a shelving rack, a table and gas bottles.
- **Pipe station:** a steel pipe (r = 0.15 m, 1.6 m long) on two saddles. This is the curved surface for force control. The robot docks head-on at its end, `cfg.pipe_work_dist` behind the work arc (Go2: 0.46 m, head 4 cm short of the pipe end; ANYmal-C: 0.56 m, nose over the low pipe end).
- **Map vs reality:** `workshop(pipe_dx, pipe_dy, pipe_dyaw_deg, pipe_radius)` moves, turns and resizes the *real* pipe. The planner and the robot only know the map (`world.map_pipes`); the real pipe (`world.pipes`) has to be sensed.
- **Beam station:** the flat beam with a straight seam (same geometry as `stance_demo.py`).
- **Geom groups:** the floor is group 0, the only thing the locomotion height scan sees (terrain). Obstacles and workpieces are group 1: the planner avoids them and the depth camera sees them. The robot is in groups 2/3. The pipe also has collision bit 2, so the force-control contact tip touches the pipe and nothing else.

`build_model(cfg, world=workshop(), contact_tip=True)` adds the world and a frictionless ball tip (r = 8 mm) at the torch tip. The old stand-alone beam (`workpiece=True`) is unchanged for `stance_demo.py`.

## Path planning (`sim/planning/`)
The map is known, so there is no SLAM.
1. **Occupancy grid:** 5 cm cells rasterised from the world's obstacle footprints. The robot is treated as a 0.5 m-radius disc (configuration space). A distance transform gives the clearance at each cell.
2. **A\*:** 8-connected. The edge cost rises within 1 m of obstacles, so paths keep a margin where there is room.
3. **Smoothing:** line-of-sight shortcuts that keep a 15 cm margin, then Chaikin corner cutting, then resampling at 5 cm. Every step is re-checked against the map.
4. **Docking:** the last 0.8 m is a straight approach along the station heading. This leg is allowed within 0.30 m of the workpiece, because the robot stands *at* it.
5. **Follower:** pure pursuit (lookahead 0.5 m, heading along the path, slows down to turn). Within 0.8 m of the goal it switches to an omnidirectional pose controller. It outputs (vx, vy, ωz) commands for the RL policy, which are the policy's native inputs. Learned policies ignore tiny commands (the fine-tuned Go2 policy stood still at 0.10 m/s, 86 mm short of its goal), so docking keeps at least 0.15 m/s until it is within 4 cm.

Start → pipe station is 7.5 m, planned in ~0.3–0.8 s on this laptop (pure Python). The minimum clearance is 0.5 m.

## Walking policies (`sim/locomotion/`)
`policy.py` runs Isaac Lab velocity-tracking policies (RSL-RL actors) in MuJoCo with the arm stowed as payload (joint PD + gravity compensation):
- **Interface:** 235-D observation at 50 Hz: base velocities, gravity, command, joint positions and velocities, last action, and a 187-ray height scan (`base_z − hit_z − 0.5`, yaw-aligned, x fastest, floor only). The scan is measured from the base height, not the +20 m sensor offset; that variant doesn't walk at all.
- **ANYmal-C** (`anymal_c`): `Isaac-Velocity-Rough-Anymal-C-v0`, ANYdrive LSTM actuator net, action scale 0.5. Tracks 0.8 → 0.75 m/s forward with the arm.
- **Go2** (`go2`): `Isaac-Velocity-Rough-Unitree-Go2-v0`, DC-motor PD (kp 25, kd 0.5, 23.5 Nm), action scale 0.25 (Isaac Lab 2.3 `UNITREE_GO2_CFG`).
  - **Joint order:** breadth-first (all hips, then thighs, then calves); depth-first falls within 0.9 s.
  - **Joint damping:** Menagerie's passive joint damping (2) and friction (0.2) are set to 0 to match Isaac Lab (`leg_passive`); with them the policy turned at only 0.22 of 0.8 rad/s.

### Fine-tuning the Go2 policy with the arm (`train_payload.py`)
```bash
python sim/locomotion/train_payload.py                                    # 800 iterations, ~24 min on this laptop (CPU)
python sim/locomotion/eval_policy.py --cmd-max 0.6 0.35 0.8 --trials 30   # compare policies
```
- **Why:** the Z1 + torch is 5.6 kg, ~37 % of a bare Go2. The pretrained policy carries it but tilts up to 18° and falls now and then.
- **Training:** PPO in MuJoCo (own implementation; 14 worker processes × 24 envs, ~11k samples/s; 6.5 M samples). It is warm-started from the pretrained Go2 actor *and* critic. Rewards are Isaac Lab's Go2 terms plus flat-orientation and base-height terms.
- **Randomisation:** arm + torch mass × U(lo, hi) with a curriculum on hi (0.6 → 1.25 over the first 150 iterations), base mass + U(−1, 2) kg, floor friction U(0.5, 1.25), velocity pushes.
- **Output:** `policies/go2/go2_z1_payload/checkpoint.pt` (RSL-RL layout, loads as loco `go2_payload`), `train_log.csv`, `train_curve.png`. The training fall rate drops from ~40 % to under 10 %.

Evaluation with the arm on board, through the same code path the mission uses (`results/locomotion/`):

| policy | forward (cmd 0.8) | yaw (0.8) | sideways (0.5) | back (0.5) | max tilt | random trials, mission envelope*: falls, velocity error (arm ×1 / ×1.25) | random trials, full ±1 range: falls (×1 / ×1.25) |
|---|---|---|---|---|---|---|---|
| `go2` pretrained | 0.74 | 0.66 | 0.41 | 0.42 | 18° | 3/30, 0.11 m/s / 1/30, 0.12 m/s | 2/20 / 1/20 |
| `go2_payload` fine-tuned | **0.80** | 0.67 | **0.47** | **0.48** | **5°** | **0/30, 0.05 m/s / 0/30, 0.05 m/s** | 2/20 / 2/20 |

\* 20 s trials, a new random command every 4 s, |vx| ≤ 0.6, |vy| ≤ 0.35, |ωz| ≤ 0.8: what the path follower commands. All remaining falls (both policies) are full-speed sideways reversals (vy −0.9 → +0.9 m/s), outside what the mission asks for.

## Perception: sensing the curved surface (`sim/perception/`)
The robot is not given the pipe; it measures it.
- **Sensor** (`depth_camera.py`): a RealSense D435-class depth camera on the Go2's head (87° × 58°, 160 × 120 rays, 0.15–3 m, noise 0.5 mm + 2 mm·z²), ray-cast with `mj_multiRay`. It sees the floor and the world, not the robot.
- **Shape fit** (`cylinder_fit.py`):
  1. Keep a region of interest around where the map roughly says the pipe is.
  2. Average 5–10 frames into 1 cm voxels.
  3. Estimate normals with a 20-neighbour PCA.
  4. RANSAC: two oriented points define a cylinder (axis = n₁ × n₂, centre where the normal lines meet).
  5. Least-squares refinement; the extent along the axis gives the near end.
  6. Sanity checks against the map before acting on the fit.
- **Viewpoint matters:** the Go2 camera is only ~4 cm above the pipe top. Head-on it sees the end cap and a grazing sliver of the top, and the fit fails. From 0.6 m to the side (`world.pipe_inspect_pose`) it worked in every tested case: radius within 1 mm, axis within 0.1°, near end within 3 mm, for pipes moved up to 8 cm, turned up to 8° and resized 0.12–0.17 m.
- **Touch** (optional, `--normal touch`): the contact tip is frictionless, so the measured contact force points exactly along the surface normal. Touch mode steers the tool axis and the arc angle from that measured normal. It works once a good first contact is made, but it cannot recover from a badly wrong model: starting from the wrong map pipe, the approach misses the real one. Vision + model was the most accurate, so it is the default.

## Force control (`sim/force_demo.py`, `sim/force_task.py`)
The torch tip presses on the pipe with a constant normal force while it sweeps a ±30° arc over the pipe top. The surface normal therefore turns through 60°, and the base is held rigid.
```bash
python sim/force_demo.py                          # Go2, 5 N, ±30°, 2 passes, depth-camera pipe -> results/force_control/
python sim/force_demo.py --pipe-dx 0.04 --pipe-dy 0.06 --pipe-dyaw 6 --pipe-radius 0.12    # real pipe differs from the map
python sim/force_demo.py --perception map         # trust the map instead of the sensor
python sim/force_demo.py --robot anymal_c_z1 --force 10 --passes 4
python sim/force_demo.py --ctrl-torch-mass 0.6 --force-noise 0.3
```
**Hybrid force/motion control inside the same QP:**
- Position is controlled in the tangent plane, and the tool axis is held on the surface normal.
- The tool's roll is left free, because a ball tip is symmetric. This is a 5-DoF task. With roll fixed, even ±30° was unreachable from the original 0.70 m stance.
- Along the normal:
  - Free space: approach at 15 mm/s.
  - In contact: admittance v_n = −k_f (F_des − F_meas), plus the desired force as a feed-forward external force in the floating-base dynamics, plus an integral of the force error (explicit PI force control).
- Feeding the *measured* force into the dynamics is positive feedback, because the soft-contact force equals the applied push. It ran away to ~200 N.

- **Contact-triggered timeline:** the sweep starts only after the tip has touched the surface; the approach keeps searching for up to 6 s more.

Results on the Go2 (5 N target, 2 passes, 30 mm/s, during the sweep). The real pipe is moved 6 cm sideways and 4 cm along, turned 6°, and is 0.12 m instead of 0.15 m in radius. All accuracy metrics are measured against the *real* pipe:

| robot believes | pipe estimate error | mean force | RMS force error | min / max | contact lost | arc on the real pipe (target ±30°) | tool axis vs real normal | base drift |
|---|---|---|---|---|---|---|---|---|
| the map | 63 mm, 6°, radius +30 mm | 4.96 N | 1.36 N | 0 / 7.6 N | 3 % | −49° … −6° | 31° RMS | 0.59 mm |
| **depth camera** | **0.2 mm, 0.01°, radius +0.2 mm** | **5.02 N** | **0.42 N** | 3.3 / 6.3 N | **0 %** | **−30.1° … +29.7°** | **1.5° RMS** | 0.60 mm |
| depth camera + touch | same | 4.96 N | 0.58 N | 1.7 / 6.3 N | 0 % | −30.5° … +29.9° | 2.1° RMS | 0.60 mm |

ANYmal-C + Z1 with the pipe as in the map (earlier run, `results/force_control/anymal_c_z1_map_model_v1.*`): 5.01 N mean, 0.27 N RMS error, 1.0° tool-axis error, 0.48 mm base drift.

The force dips and peaks briefly at the sweep's acceleration and deceleration ramps. **Reach limit:** with the 0.27 m torch, the Z1 can hold the tool normal to the pipe up to about ±35° from the top. Beyond that the flange would have to climb out of the workspace. On the Go2 the Z1 is mounted 10 cm ahead of the base centre: the Go2 is 0.2 m lower than ANYmal-C, and over the centre the Z1 only reaches ~0.5 m ahead at pipe height.

## Mission (`sim/mission_demo.py`)
```bash
python sim/mission_demo.py                     # Go2 + fine-tuned policy, real pipe moved vs the map -> results/mission/
python sim/mission_demo.py --loco go2          # pretrained Isaac Lab Go2 policy instead
python sim/mission_demo.py --robot anymal_c_z1 --no-video
```
1. Plan start → inspection pose on the map (A*).
2. Walk there with the RL policy and the path follower.
3. Stand, take 10 depth frames and fit the pipe.
4. Walk the straight docking leg to the pose computed from the *measured* pipe.
5. Hand-off to the stance WBC: brace, unstow, 5 N pass over ±30°, stow.
6. Hand back to the RL policy, plan home, walk back.

Go2 + Z1, fine-tuned policy, real pipe 63 mm / 6° / 30 mm radius off the map:

| step | result |
|---|---|
| plan + walk to the inspection pose (7.5 m) | 0.2 s planning, 24.5 s walking, within 36 mm / 0.8° |
| scan + cylinder fit (10 frames, 0.3 s) | radius 120.3 mm (real 120), axis 0.03°, axis offset 0.4 mm, near end 0.9 mm |
| dock on the measured pipe | 4.6 s, within 26 mm / 0.9° |
| force pass | 5.02 N mean, 0.41 N RMS error, contact never lost, arc −30.2° … +29.6° on the real pipe, tool 1.5° RMS from the real normal, base drift 0.63 mm |
| walk home (7.8 m) | 19.4 s, within 15 mm |
| whole walk | no robot part other than the feet touched anything; max tilt 8°; velocity tracking 0.056 m/s RMS |

Earlier ANYmal-C run (pipe as in the map, no perception): `results/mission/*_anymal_c_z1_v1.*`.

## Layout
| path | what |
|---|---|
| `sim/wbc/robots.py` | robot configs (`go2_z1`, `anymal_c_z1`) + model builder (Menagerie robot + Z1 + torch + world, torque motors) |
| `sim/wbc/stance_wbc.py` | stance WBC (QP), IK, helpers |
| `sim/stance_demo.py` | weld / stress scenarios, metrics, plots, video |
| `sim/world.py` | workshop environment (obstacles, stations) → MuJoCo scene + planner footprints |
| `sim/planning/` | occupancy grid + A* + smoothing (`grid_planner.py`), pure-pursuit / docking follower (`follower.py`) |
| `sim/locomotion/policy.py` | Isaac Lab velocity policies in MuJoCo (ANYmal-C LSTM actuator net, Go2 DC-motor PD) |
| `sim/locomotion/train_payload.py`, `eval_policy.py` | PPO fine-tuning with the arm payload (multi-process MuJoCo) and policy evaluation |
| `sim/perception/` | head depth camera (`depth_camera.py`) and RANSAC + least-squares cylinder fit (`cylinder_fit.py`) |
| `sim/force_task.py`, `sim/force_demo.py` | force-controlled pass on the pipe (task timeline, sensor, metrics, plots, video) |
| `sim/mission_demo.py` | full mission: plan → walk → dock → force control → walk home |
| `results/force_control/`, `results/mission/`, `results/locomotion/` | force-control, mission and walking-policy results |
| `locomotion/` | ETH legged_gym (ANYmal-C only) + server setup / train / export scripts; see `locomotion/README.md` |
| `tools/export_rslrl_checkpoint.py` | RSL-RL `checkpoint.pt` → TorchScript actor `policy.pt` |
| `policies/anymal/` | Isaac Lab pretrained rough-terrain checkpoints `Isaac-Velocity-Rough-Anymal-{C,D}-v0` (RL warm-start) |
| `policies/go2/go2_z1_payload/` | Go2 policy fine-tuned with the Z1 on board (checkpoint, training log and curve) |
| `results/stance_wbc/anymal_c_z1/` | WBC results |
| `scripts/setup_assets.sh` | fetches robot models into `third_party/` (git-ignored) |
| `archive/` | earlier Spot prototype (pretrained-policy sim-to-sim, Spot WBC, baselines) and Go1/Go2/A1 checkpoints; see `archive/README.md` |
