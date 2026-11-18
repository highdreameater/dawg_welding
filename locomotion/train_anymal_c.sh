#!/usr/bin/env bash
# Train ANYmal-C locomotion with ETH legged_gym exactly as the authors do (train.py --task=anymal_c_<terrain>),
# headless, then export the trained actor to TorchScript.  Quadruped only - no arm.
#
#   conda activate legged_gym
#   bash locomotion/train_anymal_c.sh rough                       # stairs / slopes / rough terrain (1500 iterations)
#   bash locomotion/train_anymal_c.sh flat
#   bash locomotion/train_anymal_c.sh rough --seed 1 --sim_device cuda:0 --rl_device cuda:0 --max_iterations 1500
#
# Output:
#   locomotion/legged_gym/logs/rough_anymal_c/<date>_<run>/model_<iter>.pt     (RSL-RL checkpoints, every 50 iters)
#   locomotion/legged_gym/logs/rough_anymal_c/exported/policies/policy_1.pt    (TorchScript actor, upstream export)
#   policies/anymal/legged_gym_anymal_c_<terrain>/                             (copy of policy + final checkpoint)
# Monitor:  tensorboard --logdir locomotion/legged_gym/logs
set -euo pipefail
TERRAIN="${1:-rough}"; shift || true
case "$TERRAIN" in rough|flat) ;; *) echo "terrain must be rough or flat"; exit 1;; esac
TASK="anymal_c_${TERRAIN}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"   # libpython for Isaac Gym

cd "$HERE/legged_gym"
echo "=== training $TASK (headless) ==="
python legged_gym/scripts/train.py --task="$TASK" --headless "$@"

echo "=== exporting latest run ==="
python "$HERE/export_policy.py" --task="$TASK" --headless "$@"
