"""Fine-tune a walking policy with the arm on board: PPO in MuJoCo (CPU, multi-process), warm-started from the
pretrained Isaac Lab Go2 rough-terrain policy (Isaac-Velocity-Rough-Unitree-Go2-v0).

    python sim/locomotion/train_payload.py                       # 600 iterations, 14 workers x 24 envs
    python sim/locomotion/train_payload.py --iters 1500 --workers 12 --envs 32
    python sim/locomotion/train_payload.py --resume              # continue from policies/go2/go2_z1_payload/

Output: policies/go2/go2_z1_payload/checkpoint.pt (RSL-RL layout: actor.*, critic.*, std -> loads in
sim/locomotion/policy.py as loco "go2_payload"), train_log.csv, train_curve.png.

Environment (same interface as the Isaac Lab velocity task, so the pretrained actor/critic carry over):
    robot      go2_z1: Go2 + Z1 (stowed, joint PD + gravity compensation) + 1.2 kg torch, Isaac-matched leg joints
    control    50 Hz policy, 200 Hz physics, target = q0 + 0.25 a, DC-motor PD (kp 25, kd 0.5, 23.5 Nm, 30 rad/s)
    obs        235-D (Isaac order) with Isaac Lab's uniform observation noise; flat floor -> height scan =
               clip(base_z - 0.5); in deployment the scan also sees only the floor (obstacles are group 1)
    commands   vx, vy, wz ~ U(-1, 1) (resampled every 10 s, 5 % standing)
    rewards    Isaac Lab Go2 rough terms (x 0.02 s): track_lin 1.5, track_ang 0.75, lin_vel_z -2, ang_vel_xy -0.05,
               torques -2e-4, dof_acc -2.5e-7, action_rate -0.01, feet_air_time 0.01 (0.5 s),
               + flat_orientation -2.0 and base_height -10 (payload: keep the base level and up)
    random     payload (arm + torch) mass x U(lo, hi) with a curriculum on hi, base mass + U(-1, 2) kg,
               floor friction U(0.5, 1.25), small velocity pushes every 6-10 s
    done       base / thigh / hip touches the floor, tilt > 60 deg, or 20 s timeout (bootstrapped)
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SIM)
sys.path.insert(0, SIM)

DT, DEC = 0.005, 4
STEP_DT = DT * DEC
EP_LEN = int(20.0 / STEP_DT)
OUT = os.path.join(ROOT, "policies", "go2", "go2_z1_payload")
W = dict(track_lin=1.5, track_ang=0.75, lin_vel_z=-2.0, ang_vel_xy=-0.05, torques=-2e-4, dof_acc=-2.5e-7,
         action_rate=-0.01, feet_air=0.01, flat_orient=-2.0, base_height=-10.0)
BASE_H_TARGET = 0.32
NOISE = np.concatenate([np.full(3, 0.1), np.full(3, 0.2), np.full(3, 0.05), np.zeros(3), np.full(12, 0.01),
                        np.full(12, 1.5), np.zeros(12), np.full(187, 0.1)])


# ====================================================================================== environments (worker side)
class EnvBatch:
    def __init__(self, n, seed, robot="go2_z1"):
        import mujoco
        from locomotion.policy import LOCO, ROBOT_LOCO
        from wbc.robots import ROBOTS, build_model
        from wbc.stance_wbc import Indexer
        self.mj = mujoco
        self.cfg = cfg = ROBOTS[robot]()
        self.lc = LOCO[ROBOT_LOCO[robot]]
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.models, self.datas = [], []
        for _ in range(n):
            m = build_model(cfg, workpiece=False, render_size=(64, 64))
            m.opt.timestep = DT
            self.models.append(m)
            self.datas.append(mujoco.MjData(m))
        m = self.models[0]
        self.ix = Indexer(m, cfg)
        J = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        self.q_idx = np.array([m.jnt_qposadr[J(nm)] for nm in self.lc.joints])
        self.v_idx = np.array([m.jnt_dofadr[J(nm)] for nm in self.lc.joints])
        self.q0 = np.array([cfg.leg_q0[nm] for nm in self.lc.joints])
        act_of_dof = {m.jnt_dofadr[m.actuator_trnid[a, 0]]: a for a in range(m.nu)}
        self.leg_act = np.array([act_of_dof[v] for v in self.v_idx])
        self.arm_q, self.arm_v = self.ix.arm_q, self.ix.arm_v
        self.arm_act = np.array([act_of_dof[v] for v in self.arm_v])
        self.arm_tau = np.array([cfg.tau_limits[nm] for nm in cfg.arm_names()])
        self.base = self.ix.base
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self.payload_bodies = [b for b, nm in enumerate(names) if nm.startswith(cfg.arm_prefix) or nm == "torch"]
        self.mass0 = m.body_mass.copy()
        self.inertia0 = m.body_inertia.copy()
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # geoms whose floor contact ends the episode: everything on the base, hips and thighs
        bad_bodies = {self.base} | {b for b, nm in enumerate(names) if nm.endswith("_hip") or nm.endswith("_thigh")}
        self.bad_geom = np.isin(m.geom_bodyid, list(bad_bodies))
        self.feet = self.ix.feet
        self.payload_hi, self.payload_lo = 1.0, 0.8
        self.last_a = np.zeros((n, 12))
        self.cmd = np.zeros((n, 3))
        self.t_cmd = np.zeros(n)
        self.t_push = np.zeros(n)
        self.ep_len = np.zeros(n, int)
        self.air = np.zeros((n, 4))
        self.in_contact = np.ones((n, 4), bool)
        self.ep_ret = np.zeros(n)
        self.ep_stats = []
        for i in range(n):
            self.reset(i)

    def set_curriculum(self, lo, hi):
        self.payload_lo, self.payload_hi = lo, hi

    def sample_cmd(self, i):
        self.cmd[i] = 0.0 if self.rng.random() < 0.05 else self.rng.uniform(-1, 1, 3)
        self.t_cmd[i] = 10.0

    def reset(self, i):
        mujoco, m, d = self.mj, self.models[i], self.datas[i]
        s = self.rng.uniform(self.payload_lo, self.payload_hi)
        m.body_mass[:] = self.mass0
        m.body_inertia[:] = self.inertia0
        m.body_mass[self.payload_bodies] *= s
        m.body_inertia[self.payload_bodies] *= s
        m.body_mass[self.base] += self.rng.uniform(-1.0, 2.0)
        m.geom_friction[self.floor, 0] = self.rng.uniform(0.5, 1.25)
        mujoco.mj_setConst(m, d)  # recompute mass-dependent constants (it overwrites d.qpos -> before the reset)
        mujoco.mj_resetData(m, d)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.q_idx] = self.q0
        d.qpos[self.arm_q] = self.cfg.arm_stow
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        d.qpos[2] -= d.geom_xpos[self.feet, 2].min() - self.cfg.foot_radius - 0.002
        mujoco.mj_forward(m, d)
        self.last_a[i] = 0.0
        self.sample_cmd(i)
        self.t_push[i] = self.rng.uniform(6, 10)
        self.ep_len[i] = 0
        self.air[i] = 0.0
        self.in_contact[i] = True
        self.ep_ret[i] = 0.0

    def obs(self, i):
        d = self.datas[i]
        R = d.xmat[self.base].reshape(3, 3)
        o = np.concatenate([R.T @ d.qvel[0:3], d.qvel[3:6], R.T @ np.array([0.0, 0.0, -1.0]), self.cmd[i],
                            d.qpos[self.q_idx] - self.q0, d.qvel[self.v_idx], self.last_a[i],
                            np.full(187, np.clip(d.qpos[2] - 0.5, -1, 1))])
        return o + NOISE * self.rng.uniform(-1, 1, o.shape)

    def all_obs(self):
        return np.stack([self.obs(i) for i in range(self.n)])

    def step(self, actions):
        mujoco, lc = self.mj, self.lc
        rew, done, tout = np.zeros(self.n), np.zeros(self.n, bool), np.zeros(self.n, bool)
        obs, final_obs = np.zeros((self.n, 235)), np.zeros((self.n, 235))
        for i in range(self.n):
            m, d, a = self.models[i], self.datas[i], actions[i]
            target = self.q0 + lc.action_scale * a
            for _ in range(DEC):
                q, qd = d.qpos[self.q_idx], d.qvel[self.v_idx]
                tau = lc.kp * (target - q) - lc.kd * qd
                hi = np.clip(lc.sat_effort * (1.0 - qd / lc.vel_limit), 0.0, lc.effort_limit)
                lo = np.clip(lc.sat_effort * (-1.0 - qd / lc.vel_limit), -lc.effort_limit, 0.0)
                tau = np.clip(tau, lo, hi)
                d.ctrl[self.leg_act] = tau
                d.ctrl[self.arm_act] = np.clip(150.0 * (self.cfg.arm_stow - d.qpos[self.arm_q]) - 8.0 * d.qvel[self.arm_v]
                                               + d.qfrc_bias[self.arm_v], -self.arm_tau, self.arm_tau)
                mujoco.mj_step(m, d)
            R = d.xmat[self.base].reshape(3, 3)
            v_b, w_b = R.T @ d.qvel[0:3], d.qvel[3:6]
            g_b = R.T @ np.array([0.0, 0.0, -1.0])
            c = self.cmd[i]
            # feet air time (Isaac Lab feet_air_time: reward the air time of a swing when the foot lands)
            contact = d.geom_xpos[self.feet, 2] < self.cfg.foot_radius + 0.006
            first = contact & ~self.in_contact[i]
            r_air = np.sum((self.air[i] - 0.5) * first) * (np.linalg.norm(c[:2]) > 0.1)
            self.air[i] = np.where(contact, 0.0, self.air[i] + STEP_DT)
            self.in_contact[i] = contact
            r = (W["track_lin"] * np.exp(-np.sum((c[:2] - v_b[:2]) ** 2) / 0.25)
                 + W["track_ang"] * np.exp(-(c[2] - w_b[2]) ** 2 / 0.25)
                 + W["lin_vel_z"] * v_b[2] ** 2 + W["ang_vel_xy"] * np.sum(w_b[:2] ** 2)
                 + W["torques"] * np.sum(tau ** 2) + W["dof_acc"] * np.sum(d.qacc[self.v_idx] ** 2)
                 + W["action_rate"] * np.sum((a - self.last_a[i]) ** 2) + W["feet_air"] * r_air
                 + W["flat_orient"] * np.sum(g_b[:2] ** 2) + W["base_height"] * (d.qpos[2] - BASE_H_TARGET) ** 2)
            rew[i] = r * STEP_DT
            self.last_a[i] = a
            self.ep_len[i] += 1
            self.ep_ret[i] += rew[i]
            # termination
            fell = g_b[2] > -0.5 or d.qpos[2] < 0.12
            if not fell:
                for k in range(d.ncon):
                    ct = d.contact[k]
                    if (ct.geom1 == self.floor and self.bad_geom[ct.geom2]) or (ct.geom2 == self.floor and self.bad_geom[ct.geom1]):
                        fell = True
                        break
            # command resampling / pushes
            self.t_cmd[i] -= STEP_DT
            if self.t_cmd[i] <= 0:
                self.sample_cmd(i)
            self.t_push[i] -= STEP_DT
            if self.t_push[i] <= 0:
                d.qvel[0:2] += self.rng.uniform(-0.3, 0.3, 2)
                self.t_push[i] = self.rng.uniform(6, 10)
            if fell or self.ep_len[i] >= EP_LEN:
                done[i], tout[i] = True, not fell
                self.ep_stats.append((self.ep_len[i], self.ep_ret[i], bool(fell)))
                final_obs[i] = self.obs(i)  # value bootstrap on timeouts uses the pre-reset observation
                self.reset(i)
                obs[i] = self.obs(i)
            else:
                obs[i] = final_obs[i] = self.obs(i)
        stats, self.ep_stats = self.ep_stats, []
        return obs, rew, done, tout, final_obs, stats


def worker(conn, n, seed):
    torch.set_num_threads(1)
    env = EnvBatch(n, seed)
    conn.send(env.all_obs())
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "curriculum":
            env.set_curriculum(*arg)
        elif cmd == "close":
            break


class VecEnv:
    def __init__(self, workers, envs, seed=0):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, envs, seed * 1000 + w), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.obs = np.concatenate([p.recv() for p in self.pipes])
        self.n, self.k = workers * envs, envs

    def step(self, actions):
        for w, p in enumerate(self.pipes):
            p.send(("step", actions[w * self.k:(w + 1) * self.k]))
        res = [p.recv() for p in self.pipes]
        obs, rew, done, tout, tout_obs = (np.concatenate([r[j] for r in res]) for j in range(5))
        stats = [s for r in res for s in r[5]]
        return obs, rew, done, tout, tout_obs, stats

    def curriculum(self, lo, hi):
        for p in self.pipes:
            p.send(("curriculum", (lo, hi)))

    def close(self):
        for p in self.pipes:
            p.send(("close", None))


# ====================================================================================== PPO (learner side)
class ActorCritic(nn.Module):
    """Same layout/names as RSL-RL's ActorCritic (actor.*, critic.*, std) so Isaac Lab checkpoints load as-is."""

    def __init__(self, n_obs=235, n_act=12, hidden=(512, 256, 128)):
        super().__init__()

        def mlp(n_out):
            layers, d = [], n_obs
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            return nn.Sequential(*layers, nn.Linear(d, n_out))

        self.actor, self.critic = mlp(n_act), mlp(1)
        self.std = nn.Parameter(torch.ones(n_act))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.std.clamp(0.05, 1.5))


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    ac = ActorCritic()
    init = os.path.join(OUT, "checkpoint.pt") if args.resume else args.init
    ck = torch.load(init, map_location="cpu", weights_only=False)
    ac.load_state_dict(ck["model_state_dict"])
    it0 = ck.get("iter", 0) if args.resume else 0
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    env = VecEnv(args.workers, args.envs, args.seed)
    N, T = env.n, args.steps
    print(f"{N} envs ({args.workers} workers x {args.envs}), {N * T} samples / iteration, init {os.path.relpath(init, ROOT)}")
    obs = torch.as_tensor(env.obs, dtype=torch.float32)
    log_f = open(os.path.join(OUT, "train_log.csv"), "a" if args.resume else "w", newline="")
    log = csv.writer(log_f)
    if not args.resume:
        log.writerow(["iter", "samples", "mean_step_rew", "ep_len", "ep_return", "fall_rate", "lr", "kl", "std",
                      "payload_hi", "sps", "wall_s"])
    lr, t_start, samples = args.lr, time.time(), 0
    for it in range(it0, it0 + args.iters):
        # payload curriculum: the upper mass scale ramps 0.6 -> 1.25 over the first `ramp` iterations
        hi = min(1.25, 0.6 + 0.65 * (it - it0 + 1) / max(1, args.ramp))
        env.curriculum(max(0.5, hi - 0.4), hi)
        t0 = time.time()
        B = dict(obs=torch.zeros(T, N, 235), act=torch.zeros(T, N, 12), logp=torch.zeros(T, N), val=torch.zeros(T, N),
                 rew=torch.zeros(T, N), done=torch.zeros(T, N))
        stats = []
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                a = dist.sample()
                B["obs"][t], B["act"][t] = obs, a
                B["logp"][t] = dist.log_prob(a).sum(-1)
                B["val"][t] = ac.critic(obs).squeeze(-1)
                o, r, dn, to, to_obs, st = env.step(a.numpy().astype(float))
                r = torch.as_tensor(r, dtype=torch.float32)
                if to.any():  # bootstrap timeouts with the value of the final (pre-reset) state
                    v_to = ac.critic(torch.as_tensor(to_obs, dtype=torch.float32)).squeeze(-1)
                    r = r + args.gamma * v_to * torch.as_tensor(to, dtype=torch.float32)
                B["rew"][t], B["done"][t] = r, torch.as_tensor(dn, dtype=torch.float32)
                obs = torch.as_tensor(o, dtype=torch.float32)
                stats += st
            last_v = ac.critic(obs).squeeze(-1)
            adv = torch.zeros(T, N)
            gae = torch.zeros(N)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else B["val"][t + 1]
                nd = 1.0 - B["done"][t]
                delta = B["rew"][t] + args.gamma * nv * nd - B["val"][t]
                gae = delta + args.gamma * args.lam * nd * gae
                adv[t] = gae
            ret = adv + B["val"]
        t_collect = time.time() - t0
        # ---- PPO update
        flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in B.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        kls = []
        for _ in range(args.epochs):
            perm = torch.randperm(T * N)
            for mb in perm.chunk(args.minibatches):
                dist = ac.dist(flat["obs"][mb])
                logp = dist.log_prob(flat["act"][mb]).sum(-1)
                ratio = torch.exp(logp - flat["logp"][mb])
                a_mb = flat["adv"][mb]
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb).mean()
                v = ac.critic(flat["obs"][mb]).squeeze(-1)
                v_old = flat["val"][mb]
                v_clip = v_old + (v - v_old).clamp(-args.clip, args.clip)
                vl = torch.max((v - flat["ret"][mb]) ** 2, (v_clip - flat["ret"][mb]) ** 2).mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kl = (flat["logp"][mb] - logp).mean().item()
                kls.append(kl)
                # adaptive learning rate (RSL-RL style, desired KL)
                if kl > 2 * args.kl:
                    lr = max(1e-5, lr / 1.5)
                elif 0 < kl < args.kl / 2:
                    lr = min(1e-3, lr * 1.5)
                for g in opt.param_groups:
                    g["lr"] = lr
        samples += T * N
        sps = T * N / t_collect
        if stats:
            el, er, fl = map(np.array, zip(*stats))
            ep_len, ep_ret, fall = el.mean() * STEP_DT, er.mean(), fl.mean()
        else:
            ep_len = ep_ret = fall = float("nan")
        row = [it + 1, samples, float(B["rew"].mean()), ep_len, ep_ret, fall, lr, float(np.mean(kls)),
               float(ac.std.mean().detach()), hi, sps, time.time() - t_start]
        log.writerow(row)
        log_f.flush()
        if (it + 1) % args.print_every == 0 or it == it0:
            print(f"it {it + 1:5d}  rew/step {row[2]:+.4f}  ep {ep_len:5.1f}s  ret {ep_ret:6.2f}  fall {fall:5.1%}  "
                  f"payload<= {hi:.2f}  lr {lr:.1e}  kl {row[7]:.4f}  std {row[8]:.3f}  {sps:6.0f} sps  "
                  f"{time.time() - t_start:6.0f}s", flush=True)
        if (it + 1) % 50 == 0 or it == it0 + args.iters - 1:
            torch.save({"model_state_dict": ac.state_dict(), "optimizer_state_dict": opt.state_dict(), "iter": it + 1,
                        "infos": {"init": os.path.relpath(init, ROOT), "robot": "go2_z1"}},
                       os.path.join(OUT, "checkpoint.pt"))
    env.close()
    log_f.close()
    plot_curve(os.path.join(OUT, "train_log.csv"), os.path.join(OUT, "train_curve.png"))


def plot_curve(csv_path, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    it = np.array([float(r["iter"]) for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, key, lab in zip(ax, ["ep_return", "fall_rate", "ep_len"], ["episode return", "fall rate", "episode length [s]"]):
        y = np.array([float(r[key]) for r in rows])
        a.plot(it, y, lw=0.5, color="0.75")
        ok = ~np.isnan(y)  # rolling mean over 25 iterations (only iterations in which episodes ended)
        k = 25
        sm = np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) if ok[max(0, i - k + 1):i + 1].any() else np.nan
                       for i in range(len(y))])
        a.plot(it, sm, lw=1.6, color="#1f6feb")
        a.set_xlabel("iteration"); a.set_title(lab); a.grid(alpha=0.3)
    fig.suptitle("Go2 + Z1 payload fine-tuning (PPO in MuJoCo)")
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ROOT, "archive", "other_robots", "policies",
                                                   "Isaac-Velocity-Rough-Unitree-Go2-v0", "checkpoint.pt"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--ramp", type=int, default=150, help="payload curriculum length [iterations]")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=24, help="rollout length per env per iteration")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--print-every", type=int, default=10)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
