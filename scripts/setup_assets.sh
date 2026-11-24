#!/usr/bin/env bash
# Fetch the robot models (MuJoCo Menagerie, sparse checkout) into third_party/ (git-ignored).
#   bash scripts/setup_assets.sh            # ANYmal-C + Unitree Go2 + Unitree Z1
#   bash scripts/setup_assets.sh --spot     # also Boston Dynamics Spot (only needed for archive/spot)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRS=(anybotics_anymal_c unitree_z1 unitree_go2)
[[ "${1:-}" == "--spot" ]] && DIRS+=(boston_dynamics_spot)

mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"
if [[ ! -d mujoco_menagerie ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
fi
cd mujoco_menagerie
git sparse-checkout add "${DIRS[@]}"
echo "menagerie models ready: ${DIRS[*]}"
