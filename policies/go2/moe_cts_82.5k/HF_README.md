---
license: mit
---

# Go2 RL Data

Checkpoint files for [go2_rl_lab](https://github.com/wty-yy/go2_rl_lab), [go2_rl_gym](https://github.com/wty-yy/go2_rl_lab/tree/gym) and [go2_rl_robotlab](https://github.com/wertyuilife2/go2_rl_robotlab).

## 📦 Files

The table below includes released checkpoints and their RoboGauge evaluation results.
`Link` points to the corresponding Hugging Face file or folder page.
`Last Updated` records the artifact release date; legacy entries without a dated name use their initial Hugging Face release date.
Scores are rounded to four decimal places from evaluation YAML files; directory names retain their original score labels.

| Model | Best Train Step | Score | Tracking | Safety | Quality | Level | Link | Last Updated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| go2_moe_cts (RobotLab v4.2) | 176k | 0.6984 | 0.7055 | 0.7850 | 0.7845 | **8.62** | [go2_moe_cts_v4.2_176k_0.6984](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_robotlab/go2_moe_cts_v4.2_176k_0.6984_20260626) | 2026-06-26 |
| go2_mixed_moe_cts (go2_rl_lab mixed1.3.2resume) | 82.5k | 0.6930 | 0.7004 | 0.7818 | 0.7818 | 8.40 | [mixed1.3.2resume_82.5k_0.6929](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929) | 2026-10-17 |
| go2_mixed_moe_cts (go2_rl_lab v1.3) | 89k | 0.6903 | 0.6950 | 0.7773 | 0.7775 | 8.34 | [v1.3_8192_89k_0.69](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_lab/2026-09-25_21-36-26_v1.3_8192_89k_0.69) | 2026-10-17 |
| go2_mixed_moe_cts (go2_rl_lab mixed1.3.2resume) | 74k | 0.6866 | 0.6944 | 0.7763 | 0.7762 | 8.34 | [mixed1.3.2resume_74k_0.687](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_74k_0.687) | 2026-10-17 |
| go2_moe_cts (RobotLab v3.3) | 185k | 0.6828 | 0.6785 | 0.7552 | 0.7645 | 8.17 | [go2_moe_cts_v3.3_185k_0.6828](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_robotlab/go2_moe_cts_v3.3_185k_0.6828_jit_onnx_robogauge_result_20260616) | 2026-06-16 |
| go2_moe_cts (go2_rl_gym) | 137k | 0.6713 | 0.6669 | 0.7857 | 0.7392 | 7.85 | [go2_moe_cts_137000_0.6713](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_moe_cts_137000_0.6713) | 2026-01-31 |
| go2_ac_moe_cts | 115k | 0.6509 | 0.6442 | 0.7644 | 0.7149 | 7.52 | [go2_ac_moe_cts_115k_0.6509.pt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_ac_moe_cts_115k_0.6509.pt) | 2026-01-31 |
| go2_mcp_cts | 91k | 0.6399 | 0.6355 | 0.7542 | 0.7058 | 7.41 | [go2_mcp_cts_91k_0.6399](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_mcp_cts_91k_0.6399) | 2026-01-31 |
| go2_moe_ng_cts | 79k | 0.6519 | 0.6447 | 0.7639 | 0.7186 | 7.56 | [go2_moe_ng_cts_79k_0.6519](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_moe_ng_cts_79k_0.6519) | 2026-01-31 |
| [CTS](https://arxiv.org/pdf/2405.10830) vanilla | 103.5k | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 | [go2_cts_vanilla2_103.5k_0.5786](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_cts_vanilla2_103.5k_0.5786) | 2026-01-31 |
| [HIM](https://arxiv.org/abs/2312.11460) | 21k | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 | [go2_him_21k_0.5379.pt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_him_21k_0.5379.pt) | 2026-01-31 |
| [DreamWaQ](https://arxiv.org/abs/2301.10602) | 119.5k | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 | [go2_dwaq_119.5k_0.5054.pt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_dwaq_119.5k_0.5054.pt) | 2026-01-31 |

Additional self-collision-enabled result:

| Model | Best Train Step | Score | Download | Last Updated |
| --- | --- | --- | --- | --- |
| go2_moe_cts_high_slope_thre | 164k | 0.6715 | [go2_moe_cts_high_slope_thre_164k_0.6715_20260419](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_moe_cts_high_slope_thre_164k_0.6715_20260419) | 2026-04-19 |

## 🚀 Usage

File types:

- `policy.pt`: TorchScript policy for Sim2Sim deployment.
- `policy.onnx`: ONNX policy for C++ / Sim2Real deployment.
- `policy.pkl`: raw policy weights.
- `model_*.pt`: full training checkpoint.
- `exported/config.yaml`: deployment configuration for `go2_rl_lab` policies.
- `params/`: saved training configuration for `go2_rl_lab` checkpoints.
- `robogauge_results_*.yaml`: RoboGauge evaluation results for `go2_rl_lab` checkpoints.
- single `*.pt` files at repository root: standalone checkpoint files kept as released.

Example paths:

```bash
# Sim2Sim policy
./go2_moe_cts_137000_0.6713/policy.pt

# ONNX deployment policy
./go2_moe_cts_137000_0.6713/policy.onnx

# Full training checkpoint
./go2_moe_cts_self_103.5k_0.6669/model_103500.pt

# go2_rl_lab ONNX deployment policy (config.yaml is in the same directory)
./go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929/exported/policy.onnx

# go2_rl_lab full training checkpoint
./go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929/model_82500.pt
```

For training, evaluation, export, and deployment commands, see the main repository [go2_rl_gym README](https://github.com/wty-yy/go2_rl_lab/tree/gym).

## 🔎 Details

- Most directory entries contain exported policies.
- Each `go2_rl_lab/<run>/` directory contains a full training checkpoint, TorchScript and ONNX policies under `exported/`, deployment and training configurations, and RoboGauge evaluation results.
- `go2_moe_cts_high_slope_thre_164k_0.6715_20260419` contains both exported policies and the full `model_164000.pt`.
- `go2_moe_cts_self_103.5k_0.6669` contains exported policies and `model_103500.pt`.
- On the Hugging Face web page, folder entries should use `tree/main/...` URLs for navigation.

## 📄  Citation
If you find our work helpful, please cite:
```bibtex
@INPROCEEDINGS{WuT-RSS-26,
    AUTHOR    = {Tianyang Wu AND Hanwei Guo AND Yuhang Wang AND Junshu Yang AND Xinyang Sui AND Jiayi Xie AND Xingyu Chen AND Zeyang Liu AND Xuguang Lan},
    TITLE     = {{Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2026.XXII.156}
}
```
