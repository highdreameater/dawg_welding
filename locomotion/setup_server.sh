#!/usr/bin/env bash
# One-time setup of the ETH legged_gym training env (ANYmal-C) on a Linux GPU server.
# Follows the upstream README (python 3.8, Isaac Gym, rsl_rl v1.0.2).
#
#   1) Download Isaac Gym (Preview 4) from https://developer.nvidia.com/isaac-gym  (needs an NVIDIA account,
#      it is not pip-installable) and extract it:   tar -xf IsaacGym_Preview_4_Package.tar.gz
#   2) bash locomotion/setup_server.sh /path/to/isaacgym          # the extracted folder (contains python/)
#
# Env vars:  ENV_NAME (default legged_gym)   TORCH=author|modern (default: auto from GPU)
set -euo pipefail

ISAACGYM_DIR="${1:?usage: bash locomotion/setup_server.sh /path/to/isaacgym}"
ENV_NAME="${ENV_NAME:-legged_gym}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -d "$ISAACGYM_DIR/python" ]] || { echo "error: $ISAACGYM_DIR/python not found (pass the extracted isaacgym folder)"; exit 1; }

# --- torch: upstream uses torch 1.10 + cu113. cu113 builds have no kernels for compute capability >= 8.9
#     (RTX 40xx, L4, L40, H100), so use torch 2.0.1 + cu118 on those GPUs.
if [[ -z "${TORCH:-}" ]]; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
  if [[ -n "$CC" ]] && awk "BEGIN{exit !($CC >= 8.9)}"; then TORCH=modern; else TORCH=author; fi
  echo "GPU compute capability: ${CC:-unknown} -> TORCH=$TORCH"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.8
conda activate "$ENV_NAME"

if [[ "$TORCH" == "author" ]]; then
  pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
else
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
fi
# isaacgym still uses np.float (removed in numpy 1.24); tensorboard is used by rsl_rl for logging
pip install "numpy==1.23.5" tensorboard ninja

# --- Isaac Gym
pip install -e "$ISAACGYM_DIR/python"

# --- rsl_rl v1.0.2 (the PPO version legged_gym was written against)
mkdir -p "$HERE/deps"
if [[ ! -d "$HERE/deps/rsl_rl" ]]; then git clone https://github.com/leggedrobotics/rsl_rl.git "$HERE/deps/rsl_rl"; fi
(cd "$HERE/deps/rsl_rl" && git checkout v1.0.2 && pip install -e .)

# --- legged_gym (vendored, ANYmal-C only). --no-deps: its install_requires names 'rsl-rl', which must not be
#     re-resolved from PyPI (we pinned v1.0.2 above).
pip install --no-deps -e "$HERE/legged_gym"
pip install matplotlib

# --- smoke test (Isaac Gym needs libpython from the conda env)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -c "import isaacgym, torch; from legged_gym.envs import task_registry; print('ok: tasks', list(task_registry.task_classes), 'cuda', torch.cuda.is_available())"
echo "setup done.  next:  conda activate $ENV_NAME && bash locomotion/train_anymal_c.sh rough"
