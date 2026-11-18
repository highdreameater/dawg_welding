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
| RL locomotion with arm payload | planned: warm-start from the Isaac Lab ANYmal rough-terrain checkpoints in `policies/anymal/`, with a curriculum that raises the arm mass step by step to the real value (cloud GPU) |
| RL ↔ WBC hand-off | planned (WBC already starts from the Isaac Lab ANYmal-C default stance) |

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

## Layout
| path | what |
|---|---|
| `sim/wbc/robots.py` | robot config + model builder (Menagerie ANYmal-C + Z1 + torch + workpiece, torque motors) |
| `sim/wbc/stance_wbc.py` | stance WBC (QP), IK, helpers |
| `sim/stance_demo.py` | weld / stress scenarios, metrics, plots, video |
| `tools/export_rslrl_checkpoint.py` | RSL-RL `checkpoint.pt` → TorchScript actor `policy.pt` |
| `policies/anymal/` | Isaac Lab pretrained rough-terrain checkpoints `Isaac-Velocity-Rough-Anymal-{C,D}-v0` (RL warm-start) |
| `results/stance_wbc/anymal_c_z1/` | WBC results |
| `scripts/setup_assets.sh` | fetches robot models into `third_party/` (git-ignored) |
| `archive/` | earlier Spot prototype (pretrained-policy sim-to-sim, Spot WBC, baselines) and Go1/Go2/A1 checkpoints; see `archive/README.md` |
