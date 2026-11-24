# dawg_welding: welding quadruped (ANYmal-C + Unitree Z1) in MuJoCo

An arm-mounted quadruped that walks to a weld site, stands still and welds. Simulation only.

```
RL locomotion (trot / stairs, arm stowed = static payload)  ->  reach goal
   ->  stance WBC holds the base rigid while the arm welds  ->  back to RL locomotion
```

| component | status |
|---|---|
| Robot model: ANYmal-C + Unitree Z1 + welding torch | done (`sim/wbc/robots.py`) |
| Stance whole-body controller (QP) | done (`sim/wbc/stance_wbc.py`) |
| Environment: workshop with obstacles + two work stations (pipe, beam) | done (`sim/world.py`) |
| Path planning on the known map (A* + smoothing, no SLAM) + path follower | done (`sim/planning/`) |
| Force control on a curved surface (hybrid force/motion in the WBC) | done (`sim/force_task.py`, `sim/force_demo.py`) |
| Walking in MuJoCo: pretrained Isaac Lab ANYmal-C rough policy, arm stowed as payload | done, sim-to-sim (`sim/locomotion/anymal_rl.py`) |
| RL ↔ WBC hand-off (walk → dock → force task → walk home) | done (`sim/mission_demo.py`) |
| RL locomotion, quadruped only (ETH legged_gym, `anymal_c_rough`) | training scripts ready (`locomotion/`), to run on a GPU server |
| RL locomotion trained with the arm payload | planned: start from the trained ANYmal-C policy, with a curriculum that raises the arm mass step by step to the real value |

## Setup
```bash
conda create -n spot_weld python=3.11 && conda activate spot_weld
pip install -r requirements.txt
bash scripts/setup_assets.sh          # MuJoCo Menagerie: anybotics_anymal_c + unitree_z1 -> third_party/
```

## Stance WBC
```bash
python sim/stance_demo.py                                   # weld + stress -> results/stance_wbc/anymal_c_z1/
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
- **Pipe station:** a steel pipe (r = 0.15 m, 1.6 m long) on two saddles. This is the curved surface for force control. The robot docks head-on at its end. The base stands 0.56 m behind the work arc, and the nose overhangs the low pipe end.
- **Beam station:** the flat beam with a straight seam (same geometry as `stance_demo.py`).
- **Geom groups:** world geoms are in group 0, which the locomotion height scan sees. The robot is in groups 2/3. The pipe also has collision bit 2, so the force-control contact tip touches the pipe and nothing else.

`build_model(cfg, world=workshop(), contact_tip=True)` adds the world and a frictionless ball tip (r = 8 mm) at the torch tip. The old stand-alone beam (`workpiece=True`) is unchanged for `stance_demo.py`.

## Path planning (`sim/planning/`)
The map is known, so there is no SLAM.
1. **Occupancy grid:** 5 cm cells rasterised from the world's obstacle footprints. The robot is treated as a 0.5 m-radius disc (configuration space). A distance transform gives the clearance at each cell.
2. **A\*:** 8-connected. The edge cost rises within 1 m of obstacles, so paths keep a margin where there is room.
3. **Smoothing:** line-of-sight shortcuts that keep a 15 cm margin, then Chaikin corner cutting, then resampling at 5 cm. Every step is re-checked against the map.
4. **Docking:** the last 0.8 m is a straight approach along the station heading. This leg is allowed within 0.30 m of the workpiece, because the robot stands *at* it.
5. **Follower:** pure pursuit (lookahead 0.5 m, heading along the path, slows down to turn). Within 0.8 m of the goal it switches to an omnidirectional pose controller. It outputs (vx, vy, ωz) commands for the RL policy, which are the policy's native inputs.

Start → pipe station is 7.5 m, planned in ~0.3–0.8 s on this laptop (pure Python). The minimum clearance is 0.5 m.

## Walking in MuJoCo (`sim/locomotion/anymal_rl.py`)
This is a sim-to-sim deployment of `policies/anymal/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt`, carrying the stowed Z1 and torch (5.6 kg).
- **Observation:** 235-D, 50 Hz. The height scan is 187 rays, done with `mj_ray` against world geoms only: `base_z − hit_z − 0.5`, yaw-aligned, x fastest.
- **Actuators:** the ANYdrive LSTM actuator net from `legged_gym` at 200 Hz, with the Isaac Lab DC-motor torque/speed clip.
- **Arm:** held stowed by joint PD plus gravity compensation.
- **Velocity tracking:** commanded 0.8 → 0.75 m/s forward, 0.5 → 0.44 m/s sideways, 0.8 → 0.58 rad/s yaw. Tilt is up to ~10° when turning with the arm aboard.
- **Height-scan origin:** the scan is measured from the base height, not the +20 m sensor offset. That variant doesn't walk at all.

## Force control (`sim/force_demo.py`, `sim/force_task.py`)
The torch tip presses on the pipe with a constant normal force while it sweeps a ±30° arc over the pipe top. The surface normal therefore turns through 60°, and the base is held rigid.
```bash
python sim/force_demo.py                          # 5 N, ±30°, 2 passes -> results/force_control/
python sim/force_demo.py --force 10 --passes 4
python sim/force_demo.py --ctrl-torch-mass 0.6    # wrong torch-mass belief
python sim/force_demo.py --force-noise 0.3        # noisy force sensor
```
**Hybrid force/motion control inside the same QP:**
- Position is controlled in the tangent plane, and the tool axis is held on the surface normal.
- The tool's roll is left free, because a ball tip is symmetric. This is a 5-DoF task. With roll fixed, even ±30° was unreachable from the original 0.70 m stance.
- Along the normal:
  - Free space: approach at 15 mm/s.
  - In contact: admittance v_n = −k_f (F_des − F_meas), plus the desired force as a feed-forward external force in the floating-base dynamics, plus an integral of the force error (explicit PI force control).
- Feeding the *measured* force into the dynamics is positive feedback, because the soft-contact force equals the applied push. It ran away to ~200 N.

Results (5 N target, 2 passes, 30 mm/s, during the sweep):

| mean force | RMS error | min / max | contact lost | tangential error RMS / max | tool-axis error | base drift | foot slip | peak torque |
|---|---|---|---|---|---|---|---|---|
| 5.01 N | 0.27 N | 3.4 / 5.9 N | 0 % | 1.1 / 3.8 mm | 1.0° RMS | 0.48 mm | 0.07 mm | 40 % |

The force dips by up to 1.6 N and peaks by up to 0.9 N, briefly, at the sweep's acceleration and deceleration ramps. **Reach limit:** with the 0.27 m torch, the Z1 can hold the tool normal to the pipe up to about ±35° from the top. Beyond that the flange would have to climb out of the workspace.

## Mission (`sim/mission_demo.py`)
```bash
python sim/mission_demo.py              # -> results/mission/ (video, map.png, summary.json); ~1 min without video
```
1. Plan start → pipe station.
2. The RL policy and path follower walk it and dock.
3. Hand-off to the stance WBC: brace, unstow, 5 N force pass, stow.
4. Hand back to the RL policy, plan home, walk back.

| leg | result |
|---|---|
| walk to pipe (7.5 m) | 18 s, docked within 38 mm / 2.1° |
| force pass | 5.01 N mean, 0.28 N RMS error, contact never lost, base drift 0.51 mm |
| walk home (7.5 m) | 21 s, docked within 30 mm |
| whole walk | no robot part other than the feet touched any obstacle; min clearance 0.5 m |

## Layout
| path | what |
|---|---|
| `sim/wbc/robots.py` | robot config + model builder (Menagerie ANYmal-C + Z1 + torch + workpiece, torque motors) |
| `sim/wbc/stance_wbc.py` | stance WBC (QP), IK, helpers |
| `sim/stance_demo.py` | weld / stress scenarios, metrics, plots, video |
| `sim/world.py` | workshop environment (obstacles, stations) → MuJoCo scene + planner footprints |
| `sim/planning/` | occupancy grid + A* + smoothing (`grid_planner.py`), pure-pursuit / docking follower (`follower.py`) |
| `sim/locomotion/anymal_rl.py` | pretrained Isaac Lab ANYmal-C rough policy + LSTM actuator net running in MuJoCo |
| `sim/force_task.py`, `sim/force_demo.py` | force-controlled pass on the pipe (task timeline, sensor, metrics, plots, video) |
| `sim/mission_demo.py` | full mission: plan → walk → dock → force control → walk home |
| `results/force_control/`, `results/mission/` | force-control and mission results |
| `locomotion/` | ETH legged_gym (ANYmal-C only) + server setup / train / export scripts; see `locomotion/README.md` |
| `tools/export_rslrl_checkpoint.py` | RSL-RL `checkpoint.pt` → TorchScript actor `policy.pt` |
| `policies/anymal/` | Isaac Lab pretrained rough-terrain checkpoints `Isaac-Velocity-Rough-Anymal-{C,D}-v0` (RL warm-start) |
| `results/stance_wbc/anymal_c_z1/` | WBC results |
| `scripts/setup_assets.sh` | fetches robot models into `third_party/` (git-ignored) |
| `archive/` | earlier Spot prototype (pretrained-policy sim-to-sim, Spot WBC, baselines) and Go1/Go2/A1 checkpoints; see `archive/README.md` |
