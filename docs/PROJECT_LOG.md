# dawg_welding: project log

Goal: an arm-mounted quadruped that walks to a weld site, stands still and welds, in simulation only, never on real hardware. It is inspired by the Boston Dynamics Spot welding demo.

Architecture (decided early, unchanged since):

```
RL locomotion (trot / stairs, arm folded = static payload)  ->  reach goal
   ->  stance whole-body controller (WBC) holds the base rigid while the arm welds  ->  back to RL
```

Everything below happened on 2026-11-18, in this order. All simulation runs in MuJoCo 4.0.1, conda env `spot_weld` (Python 3.11).

---

## Phase 1: Boston Dynamics Spot + Spot arm (archived in `archive/spot/`)

### 1.1 Models
- **Official description:** `bdaiinstitute/spot_description` (ROS 2 URDF/xacro, `arm:=True`).
- **Simulation model:** MuJoCo Menagerie `boston_dynamics_spot/spot_arm.xml`, which already includes the 6-DOF arm and gripper, so no URDF conversion was needed. Spot is 50.3 kg and the arm 6.0 kg.
- **Real-robot reference:** `boston-dynamics/spot-rl-example` (RL Researcher Kit deploy code) was used to get the observation layout and joint order.

### 1.2 Pretrained locomotion policies found
- **NVIDIA Isaac Sim `spot_policy.pt`:** flat terrain. The same file is served for Isaac Sim 4.2, 4.5 and 5.1.
- **NVIDIA Isaac Lab `Isaac-Velocity-Flat-Spot-v0`:** RSL-RL checkpoint, exported to TorchScript with `tools/export_rslrl_checkpoint.py`.
- **Not available:** no pretrained rough-terrain or stairs policy exists for Spot. The Hugging Face community checkpoint returned 401.

### 1.3 Zero-shot sim-to-sim in MuJoCo (`archive/spot/spot_mujoco_deploy.py`)
Interface: 48-D obs, Isaac joint order, target = q0 + 0.2·a, PD 60/1.5, 50 Hz.
- **Plain Spot, flat ground:** the transfer works, with about 0.16 m/s velocity-tracking error.
- **With the arm:** roll and pitch rise to 15–20°. The Isaac Sim policy falls during a welding-like arm sweep.
- **Stairs and rough terrain:** no policy climbs 10 cm stairs, and with the arm even 6 cm rough terrain fails.
- **Conclusion:** training is needed.

### 1.4 Stance WBC on Spot (first version)
- **Design:** inverse-dynamics QP at 500 Hz (see the WBC section below). Dynamics come from MuJoCo itself, on a separate controller model copy.
- **Bugs fixed during development:**
  - A step change in the seam reference caused single-tick torque saturation. It was replaced by a constant-speed profile with smooth (min-jerk) ramps.
  - A posture task conflicted with the 6-D tool task on the 6-DOF arm, so its weight was reduced.
- **Result:** weld base drift 0.29 mm, tilt 0.004°, tool error 0.015 mm RMS. Stress (fast sweeps): 0.29 mm, 0.013°.
- **Baselines (unrequested; kept in the archive):** WBC vs frozen-leg PD and the RL policy standing. Two fairness fixes were needed: implicit PD for stiff servos, and replanning the arm from the measured base pose. The WBC was 1–3 orders of magnitude better.

### 1.5 Why Spot was dropped
No open stairs or rough-terrain policy exists for Spot, and leading labs don't publish Spot training curricula. ANYmal and Unitree have far more open RL resources.

---

## Phase 2: ANYmal-C + Unitree Z1 (current main platform, `sim/`)

### 2.1 Model (`sim/wbc/robots.py`, config `anymal_c_z1`)
- **Robot:** Menagerie `anybotics_anymal_c` with Menagerie `unitree_z1` attached via `MjSpec` at base (0.22, 0, 0.09), plus a 1.2 kg welding torch.
- **Mass:** 50.6 kg in total, of which arm plus torch is 5.6 kg.
- **Actuators:** all are torque motors: ANYmal 80 Nm, Z1 30 Nm (joint2 60 Nm).
- **Model fixes:**
  - Z1 self-contacts are excluded, because the folded pose touches itself.
  - The Z1 stow pose was moved to `[0, 0.05, -0.05, 0, 0, 0]`, because the zero pose sits on the joint2/joint3 limits.

### 2.2 Stance WBC port: issues found and fixed
| issue | cause | fix |
|---|---|---|
| seam unreachable (113 mm IK residual) | bad IK seeds parked the arm on joint limits | multi-start IK (config seeds + 20 random), pick the exact solution with the largest joint-limit margin; workpiece moved to a narrow beam at x = 0.70 m (Z1 reach ~0.7 m) |
| 2–3° base rotation spikes | joint-limit forces from the stow pose (not in the WBC model) | stow pose inside limits; stress sweeps clipped to the joint ranges |
| 3.6 mm tool error | Z1 Coulomb friction (1 Nm per joint) causing stick-slip | friction feed-forward on the planned joint velocity, a clamped integral term on the tool task, and stiffer tool gains (kp 2500, kd 100, ki 8000) |

### 2.3 Results
| scenario | base drift | tilt | tool error RMS / max | peak torque |
|---|---|---|---|---|
| weld | 0.40 mm | 0.041° | 0.17 / 0.61 mm | 68% |
| stress | 0.39 mm | 0.022° | – | 45% |
| weld, controller's torch mass 0.6 kg vs real 1.2 kg | 0.41 mm | 0.125° | 0.32 / 1.09 mm | 55% |

### 2.4 Repo and RL training pipeline
- **Repo:** cleaned up and restructured around ANYmal + Z1, and pushed to `github.com/highdreameater/dawg_welding`.
  - `c5adc73`: restructure. All Spot material went to `archive/`, and the Go1/Go2/A1 checkpoints to `archive/other_robots/`.
  - `8bec658`: added `locomotion/legged_gym`, a copy of ETH `leggedrobotics/legged_gym` @ `8fa29ac` trimmed to ANYmal-C. The code was audited first (no network, shell or pickle calls).
- **GPU-server scripts:**
  - `setup_server.sh`: Python 3.8, Isaac Gym, rsl_rl v1.0.2. It picks the torch build from the GPU's compute capability.
  - `train_anymal_c.sh`: the authors' `train.py --task=anymal_c_rough --headless`.
  - `export_policy.py`: headless TorchScript export.
- **Not run yet:** this needs Linux and an NVIDIA GPU.

---

## Phase 3: Unitree Go2 + Z1 (`go2_z1/` (later moved to the top level), plus `sim/` config `go2_z1`)

### 3.1 Combined description
- **No official combined model:** Unitree publishes no Go2+Z1 model.
- **What was built:** `description/go2_z1.urdf` is assembled from the official `unitree_ros` `go2_description` and `z1_description` (commit `157ff95`). It uses Unitree's own mount rule from `aliengoZ1_description`: x = trunk_length/2 − 0.07, z = trunk_height/2, which gives **(0.1181, 0, 0.057)** for Go2.
- **Result:** 18 actuated joints, 20.7 kg in the URDF, Z1 4.42 kg. Validated in MuJoCo.

### 3.2 Pretrained Go2 weights (`policies/go2/`)
- **Unitree:** ships no Go2 weights (`unitree_rl_gym` has G1/H1 only).
- **Collected:**
  - wty-yy MoE-CTS (`go2_rl_lab` 82.5k, `robot_lab` symmetry 77k; MIT, RoboGauge-benchmarked).
  - HIMLoco and robot_lab via rl_sar.
  - NVIDIA Isaac Lab Go2 rough and flat.
  - m3 Go2+Z1 walking v1/v2 (arm folded, flat ground).
  - Genesis sim-to-real walk and stairs.
- **MoE-CTS papers:**
  - Wu et al., *Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion*, RSS 2026, arXiv 2602.00678.
  - Base method: Wang et al., *CTS: Concurrent Teacher-Student RL for Legged Locomotion*, arXiv 2405.10830.

### 3.3 Zero-shot locomotion in MuJoCo (`sim/locomotion_demo.py`, Menagerie Go2)
- **Interfaces:** reverse-engineered from the exported TorchScript and configs. The MoE-CTS exports keep a 10-step, term-major history internally, primed after reset. HIMLoco uses a 6-step, newest-first history.
- **Results:**

| policy | walk | stairs 10 cm (up, platform, down) | stairs 15 cm |
|---|---|---|---|
| robot_lab symmetry 77k | 0.09 m/s err | done | done |
| go2_rl_lab MoE-CTS 82.5k | 0.11 m/s | done | done |
| HIMLoco | 0.14 m/s | up only (out of time) | fails |

- **With the Z1 mounted** (folded, its own servos holding it, +4.42 kg = +29%, about 4× the training mass randomisation), both MoE policies still walk, turn and climb the 10 cm stairs with no falls. Tracking error rises about 10–25% and the body sits 2.3 cm lower.
- **One policy for everything:** the same network handles flat ground, turning and stairs; there is no mode switch. It is blind (proprioception only), so it reacts to steps on contact.
- **Step height:** we pick the step height, and 10 cm is sufficient.

### 3.4 Stance WBC on Go2 + Z1 (no torch: massless tool point 5 cm past the Z1 flange)
- **Workpiece:** 12 cm-high beam at x = 0.42 m, from a reach map. Seam ±0.10 m with ±4 mm weave. The body is lowered 3 cm to brace.
- **Issues found and fixed:**
  - **2.2 mm tool error and torque spikes:** Z1 friction stick-slip again. Confirmed: with Z1 friction removed from plant and controller, the error is 0.045 mm.
  - **Wrist-roll runaway into its limit:** with no torch, the last Z1 link is almost inertia-free and Menagerie gives the Z1 no rotor inertia. Fix: 0.01 kg·m² armature on the Z1 joints, the same value Menagerie uses for the Go2 leg motors.
  - **Weld precision:** an arm disturbance observer (`dob_T` = 4 ms) estimates and cancels unmodelled joint torques while the tool is tracked in Cartesian space. Applying it always made the stress test worse, so it is weld-only.
- **Results:**

| scenario | base drift | tilt | tool error RMS / max | peak torque |
|---|---|---|---|---|
| weld | 0.40 mm | 0.030° | **0.08 / 0.18 mm** | 35% |
| stress (fast ±1 rad arm sweeps) | 1.5 mm | 0.80° | – | 38% |

- **Stress tilt:** this is mostly physical: a 4.4 kg arm swinging on a 15 kg robot with a small support polygon.
- **No regression:** ANYmal-C + Z1 results were re-checked after the shared-controller changes and are identical.

---

## The stance WBC (shared by all robots)
`sim/wbc/stance_wbc.py`. It is solved every physics tick (500 Hz) with DAQP, in about 0.2 ms.
- **Variables:** joint accelerations and 3-D forces at the four feet.
- **Cost:**
  - Rigid base pose (PD to a desired acceleration).
  - 6-D tool pose: PD, feed-forward and a clamped integral.
  - Arm posture or joint trajectory.
  - Small regularisation terms.
- **Hard constraints:**
  - Floating-base dynamics: the unactuated rows of M q̈ + h = Sᵀτ + J_cᵀf.
  - Stance feet do not accelerate.
  - Friction pyramids (μ = 0.6, f_z ≥ 5 N).
  - Joint torque limits.
- **Model terms:** h includes passive joint damping. Coulomb friction gets a feed-forward term, or the arm disturbance observer during welding.
- **Model choice:** dynamics come from MuJoCo (`mj_fullM`, `qfrc_bias`, `mj_jac`, `mj_jacDot`) on the controller's own model copy. That copy can differ from the plant, for example a wrong torch mass.
- **Weld sequence:** settle, brace (lower the body), unfold the arm in joint space, descend in Cartesian space, weld at constant speed with weave, retract.

## Phase 4: focus on Unitree Go2 + Z1 (repo cleanup)
Decision: the project continues with **Go2 + Z1 only**. Other arms may be tried later, so arm swapping stays easy.
- `sim/wbc/robots.py` keeps the arm as config fields (`arm_xml`, mount, joints, limits).
- `scripts/setup_assets.sh --arms` fetches other small Menagerie arms (AgileX Piper, ARX L5, Kinova Gen3).

**Removed** (still recoverable from git history, `c5adc73` / `8bec658`, or the original download sources):
- the ANYmal config, the legged_gym training pipeline and the ANYmal checkpoints;
- the Spot deploy code, policies and local clones;
- the Go1/Go2/A1 Isaac Lab checkpoints, and the other Go2 weights (HIMLoco, rl_sar robot_lab, NVIDIA Isaac Lab Go2, m3 Go2+Z1, Genesis);
- the RSL-RL export tool.

**Kept:**
- the Go2 + Z1 description;
- the two final Go2 policies (MoE-CTS 82.5k, symmetry 77k) with every locomotion video;
- the Go2 stance WBC and its videos;
- in `archive/`, 1–2 videos and the result tables per earlier experiment (Spot pretrained locomotion, Spot WBC, Spot WBC vs baselines, ANYmal WBC).

**Checked after the restructure:** the WBC and locomotion demos reproduce their numbers exactly.

## Where things are
| what | where |
|---|---|
| Go2 + Z1 URDF | `description/go2_z1.urdf` (+ `build_go2_z1_urdf.py`; official parts fetched into `description/upstream/`) |
| Go2 policies | `policies/go2/{moe_cts_82.5k,symmetry_77k}` |
| Locomotion demo | `python sim/locomotion_demo.py --scenario {walk,turn,stairs} [--arm]` → `results/locomotion/` |
| Stance WBC + robot config | `sim/wbc/stance_wbc.py`, `sim/wbc/robots.py` (`go2_z1`) |
| WBC demo | `python sim/stance_demo.py [--scenario weld\|stress] [--torch-mass 1.2]` → `results/stance_wbc/go2_z1/` |
| Assets | `bash scripts/setup_assets.sh [--arms]` |
| Earlier experiments | `archive/` (see `archive/README.md`) |

## Open items / next steps
1. **Locomotion fine-tuning:** warm-start from MoE-CTS / symmetry (their training code is public: `wty-yy/go2_rl_lab`, `wertyuilife2/go2_rl_robotlab`). Use a curriculum that ramps the arm mass up to the real value, plus CoM randomisation, on the cloud GPU.
2. **RL ↔ WBC hand-off:** stop and settle, blend leg commands over about 0.5 s, then the WBC takes over, and the reverse afterwards. Both already share the policies' default stance.
3. **Torch:** add it back with its real mass once chosen (`--torch-mass`).
4. **Other arms (optional):** add a config in `robots.py` and check reach, mass and WBC margins on the Go2.

## Phase 5: full mission test, walk → stairs → stop → weld a wall crack (`sim/mission_demo.py`)
- **Course:** 3 m flat, 6 × 10 cm stairs, 3 m platform, and a wall on the left with a zig-zag crack.
- **RL phase:** the pretrained policy drives at 0.5 m/s with simple heading and lane-keeping feedback (wz = −2·yaw − 1.5·y). It stops at x = 6 m and waits for |v| < 0.05 m/s.
- **Hand-over:** blended over 0.5 s. The RL leg PD targets are frozen at the moment of hand-over, and the arm hold torque is faded into the WBC torques.
- **WBC phase:** brace, unfold, approach, weld 8 s at constant speed, retract, re-stow.
- **Problems found while setting up the wall weld:**
  - At first the crack was out of reach, at 0.26–0.44 m above the platform. A Monte Carlo workspace scan showed that a sideways (horizontal) tool reaches only about 0.45–1.0 m above the platform.
  - With the wall at 0.42 m, the elbow would have to fold to its limit. The wall was moved to 0.58 m.
  - The IK over-constrained tool roll, which is irrelevant for welding. `ik_arm(..., axis_only=True)` now constrains only the tool axis, with steps capped at 0.2 rad. Seeds come from sampled arm poses whose tool already points at the wall.
- **Result:** both policies complete the whole mission with no falls. Weld tool error 0.017 mm RMS; the body moves 0.3 mm and tilts 0.13°; the hand-over jump is 0.4–1.0 mm.
- **Videos:** `results/mission/`.
