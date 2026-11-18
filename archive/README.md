# Archive

Earlier work, kept for reference. The project now uses **ANYmal-C + Unitree Z1** only (see the top-level README).

## `spot/`: Boston Dynamics Spot + Spot arm (first prototype)
| path | what |
|---|---|
| `spot_mujoco_deploy.py` | zero-shot sim-to-sim of pretrained Spot locomotion policies in MuJoCo (`python archive/spot/spot_mujoco_deploy.py --all`, needs `bash scripts/setup_assets.sh --spot`) |
| `spot_robot_config.py` | Spot + arm config for the stance WBC |
| `policies/isaacsim_spot_flat/` | NVIDIA Isaac Sim `spot_policy.pt` + `spot_env.yaml` |
| `policies/isaaclab_flat_spot_rslrl/` | Isaac Lab `Isaac-Velocity-Flat-Spot-v0` RSL-RL checkpoint (20k iters) + exported `policy.pt` |
| `results/pretrained_locomotion/` | 20 videos + table: flat Spot policies on Spot / Spot+arm (stand, walk, arm reach/sweep, rough, stairs) |
| `results/stance_wbc/` | stance WBC on Spot + arm (weld + stress) |
| `results/stance_wbc_vs_baselines/` | WBC vs frozen-leg PD and RL-standing baselines |
| `third_party/` (git-ignored) | local clones of `bdaiinstitute/spot_description` and `boston-dynamics/spot-rl-example` |

Spot policy interface: 50 Hz, 48-D obs (base lin/ang vel, projected gravity, cmd, q−q0, q̇, last action), Isaac joint order
`fl_hx fr_hx hl_hx hr_hx fl_hy fr_hy hl_hy hr_hy fl_kn fr_kn hl_kn hr_kn`, target = q0 + 0.2·a, PD kp 60 / kd 1.5.

Pretrained-policy results (zero-shot):

| policy | robot | stand | walk | arm_reach | arm_sweep | rough (≤6 cm) | stairs (10 cm) |
|---|---|---|---|---|---|---|---|
| isaacsim | spot | ok | ok (0.16 m/s err) | – | – | fail 3.1 s | fail 4.6 s |
| isaacsim | spot_arm | ok | ok (0.20) | ok (0.28) | **fail 6.9 s** | fail 2.9 s | fail 3.8 s |
| isaaclab | spot | ok | ok (0.18) | – | – | ok (0.15) | fail 5.1 s |
| isaaclab | spot_arm | ok | ok (0.22) | ok (0.26) | ok (0.36), 18° pitch | fail 2.8 s | fail 4.0 s |

- On flat ground the policies transfer fine.
- The arm raises roll and pitch to 15–20°.
- No Spot policy climbs stairs, and no pretrained rough-terrain Spot policy exists. This is why the project moved to ANYmal.

Stance WBC on Spot + arm:
- Weld: base drift 0.29 mm, tilt 0.004°, torch error 0.015 mm RMS.
- Stress: base drift 0.29 mm, tilt 0.013°.

## `other_robots/policies/`
Isaac Lab rough-terrain RSL-RL checkpoints for Unitree A1, Go1 and Go2 (`Isaac-Velocity-Rough-*-v0`). They are from NVIDIA's asset server:
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/<task>/checkpoint.pt`
