#!/usr/bin/env bash
# Fetch external robot models (git-ignored):
#   third_party/mujoco_menagerie/{unitree_go2,unitree_z1,anybotics_anymal_c} MuJoCo models used by sim/
#   description/upstream/unitree_ros/robots/{go2,z1,aliengoZ1}_description
#                                                                 official Unitree descriptions used by description/go2_z1.urdf
#
#   bash scripts/setup_assets.sh            # Go2 + Z1 + ANYmal-C
#   bash scripts/setup_assets.sh --arms     # also other small Menagerie arms, for trying a different arm on the Go2
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MENAGERIE_DIRS=(unitree_go2 unitree_z1 anybotics_anymal_c)
[[ "${1:-}" == "--arms" ]] && MENAGERIE_DIRS+=(agilex_piper arx_l5 kinova_gen3)
UNITREE_ROS_COMMIT=157ff950351e5f535e03c5d85a16b7d594b3dbcc  # commit description/go2_z1.urdf was built from

mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"
[[ -d mujoco_menagerie ]] || git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
(cd mujoco_menagerie && git sparse-checkout add "${MENAGERIE_DIRS[@]}")

mkdir -p "$ROOT/description/upstream"
cd "$ROOT/description/upstream"
if [[ ! -d unitree_ros ]]; then
  git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git
  (cd unitree_ros && git checkout -q "$UNITREE_ROS_COMMIT")
fi
(cd unitree_ros && git sparse-checkout set robots/go2_description robots/z1_description robots/aliengoZ1_description)
echo "assets ready: menagerie ${MENAGERIE_DIRS[*]} + unitree_ros @ ${UNITREE_ROS_COMMIT:0:7}"
