"""Export a trained legged_gym ANYmal-C policy to TorchScript (headless).

Same steps as upstream legged_gym/scripts/play.py (load the latest run with resume=True, export_policy_as_jit),
without the visualisation loop, so it runs on a display-less GPU server. Also copies the exported policy and the
final RSL-RL checkpoint into <repo>/policies/anymal/legged_gym_<task>/.

    python locomotion/export_policy.py --task=anymal_c_rough --headless [--load_run <run dir> --checkpoint <iter>]
"""
import os
import shutil

import isaacgym  # noqa: F401  (must be imported before torch)
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  (registers the tasks)
from legged_gym.utils import export_policy_as_jit, get_args, task_registry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # small env, as in play.py - only needed to build the runner
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    args.num_envs = 1
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root="default")

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    path = os.path.join(log_root, "exported", "policies")
    export_policy_as_jit(runner.alg.actor_critic, path)
    print("exported TorchScript policy to", path)

    out = os.path.join(REPO_ROOT, "policies", "anymal", f"legged_gym_{args.task}")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(path, "policy_1.pt"), os.path.join(out, "policy.pt"))
    # the checkpoint that was loaded (latest run / latest model unless --load_run/--checkpoint given)
    from legged_gym.utils.helpers import get_load_path
    ckpt = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    shutil.copy(ckpt, os.path.join(out, os.path.basename(ckpt)))
    with open(os.path.join(out, "SOURCE.txt"), "w") as f:
        f.write(f"task: {args.task}\ncheckpoint: {os.path.relpath(ckpt, REPO_ROOT)}\n"
                f"obs dim: {env.num_obs}  actions: {env.num_actions}\n"
                f"policy dt: {env.dt} s  action_scale: {env_cfg.control.action_scale}\n")
    print("copied policy + checkpoint to", out)


if __name__ == "__main__":
    export(get_args())
