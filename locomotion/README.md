# ANYmal-C locomotion training (ETH legged_gym)

`legged_gym/` is a trimmed copy of [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) (BSD-3, ETH Zurich / NVIDIA), upstream commit `8fa29ac` (2025-05-29).
- **Kept** (unchanged): base env, ANYmal-C env + rough/flat configs, utils, `train.py`, `play.py`, `test_env.py`, the ANYmal-C URDF + meshes, and the ANYdrive SEA actuator network `anydrive_v3_lstm.pt`.
- **Removed:** A1, ANYmal-B and Cassie envs and assets.
- **Only edit:** `legged_gym/envs/__init__.py` registers just `anymal_c_rough` and `anymal_c_flat`.

This trains the **quadruped only**, with no arm. It is the authors' setup:
- PPO, 4096 parallel robots, and a terrain curriculum (slopes, stairs, rough terrain, stepping stones).
- Base-mass randomisation of ±5 kg.
- A 187-point height scan in the observation (235-D obs for `rough`).
- 1500 iterations.

## On the GPU server
```bash
git clone https://github.com/highdreameater/dawg_welding.git && cd dawg_welding

# 1. Isaac Gym is not on pip: download "Isaac Gym Preview 4" from https://developer.nvidia.com/isaac-gym, then
tar -xf IsaacGym_Preview_4_Package.tar.gz            # -> ./isaacgym

# 2. one-time env setup (conda env 'legged_gym': python 3.8, torch, isaacgym, rsl_rl v1.0.2, this legged_gym)
bash locomotion/setup_server.sh ./isaacgym

# 3. train + export (headless)
conda activate legged_gym
bash locomotion/train_anymal_c.sh rough
```

- **Monitoring:** `tensorboard --logdir locomotion/legged_gym/logs`
- **Extra arguments:** these pass through to `train.py`: `--seed`, `--max_iterations`, `--num_envs`, `--sim_device cuda:0`, `--rl_device cuda:0`, `--run_name`.
- **Resuming:** `--resume --load_run <run dir> --checkpoint <iter>`

### Outputs
| file | what |
|---|---|
| `locomotion/legged_gym/logs/rough_anymal_c/<date>_<run>/model_<iter>.pt` | RSL-RL checkpoints, saved every 50 iterations |
| `locomotion/legged_gym/logs/rough_anymal_c/exported/policies/policy_1.pt` | TorchScript actor (upstream `export_policy_as_jit`) |
| `policies/anymal/legged_gym_anymal_c_rough/` | copy of `policy.pt` + final checkpoint + `SOURCE.txt` (obs dim, dt, action scale), ready to commit back to this repo |

`logs/` is git-ignored. To bring the weights back, commit `policies/anymal/legged_gym_anymal_c_rough/` and push.

### Notes
- **torch version:** the setup script follows upstream (torch 1.10 + cu113). GPUs with compute capability ≥ 8.9 (RTX 40xx, L4, L40, H100) have no cu113 kernels, so on those it installs torch 2.0.1 + cu118 instead. You can force a choice with `TORCH=author` or `TORCH=modern`.
- **libpython error:** `libpython3.8.so.1.0: cannot open shared object file` is fixed by the `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` export, which both scripts already do.
- **Default stance:** the default joint pose (`HFE ±0.4`, `KFE ∓0.8`) is the same stance the stance WBC in `sim/` starts from, which keeps the later RL → WBC hand-off clean.
