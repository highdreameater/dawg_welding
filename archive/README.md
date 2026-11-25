# Archive

Earlier experiments, kept as a record: 1–2 videos and the result tables for each. The code, policies and full video sets were removed on 2026-11-18, when the project moved to Unitree Go2 + Z1. They are still in git history: commit `c5adc73` has the Spot material and the ANYmal WBC, and `8bec658` adds the ANYmal legged_gym training pipeline. Details are in `docs/PROJECT_LOG.md`.

## `spot/`: Boston Dynamics Spot + Spot arm (first prototype)
| folder | experiment | kept |
|---|---|---|
| `pretrained_locomotion/` | NVIDIA flat-terrain Spot policies (Isaac Sim / Isaac Lab), zero-shot in MuJoCo | `spot_arm__isaacsim__arm_sweep.mp4` (falls during a welding-like arm sweep), `spot_arm__isaaclab__stairs.mp4` (cannot climb 10 cm stairs), `summary.md` (all 20 runs) |
| `stance_wbc/` | first stance WBC, Spot + arm + 1.2 kg torch | `weld.mp4`, `stress.mp4`, `weld.png`, `summary.md` (weld: 0.29 mm drift, 0.015 mm tool error) |
| `wbc_vs_baselines/` | WBC vs frozen-leg PD (soft / stiff) and the RL policy standing | `weld__comparison.mp4`, `stress__comparison.mp4` (4-panel), `weld.png`, `summary.md` |

Why Spot was dropped: no open stairs or rough-terrain policy exists for it, and few open RL resources.

## `anymal_c_z1/`: ANYmal-C + Unitree Z1 (second platform)
| folder | experiment | kept |
|---|---|---|
| `stance_wbc/` | stance WBC, ANYmal-C + Z1 + 1.2 kg torch | `weld.mp4`, `stress.mp4`, `weld.png`, `summary.md`, `summary_wrong_torch_mass.md` (weld: 0.40 mm drift, 0.17 mm tool error) |

Not kept: the ETH legged_gym ANYmal-C training pipeline (prepared but never run) and the Isaac Lab ANYmal-C/D checkpoints.
