#!/usr/bin/env python3
"""Exploration-economics assay: which selection-pressure change breaks
satisficing? 2x2 matrix {fitness: late vs q4-only} x {arms: uniform vs
needle}, bred with toy-scale ES (the optimizer that settled = the most
sensitive detector). Fifth specimen: the R2 PPO session-1 agent, same
forgiving world, gradient credit assignment.

Wanting-to-explore metrics (from the fossil forensics):
  arms25  — distinct arms tried in first 25 pulls (8 = full sweep)
  P(rep|L)— P(repeat action | last pull lost); settlers stay anyway
  gate_q4 — final-quarter best-arm rate (did wanting pay?)
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from changeling import N_ACT, OBS_DIM
from changeling.agent import gru_step, hidden_size, init_gru
from changeling.envs import bandit_env
from changeling.ppo import init_ppo_params
from changeling.train import train

T, K = 200, 8
Q4 = T - T // 4
TOY = dict(env="bandit", gens=800, hidden=32, pop=128, n_lifetimes=4,
           sigma=0.03, lr=0.02, seed=0, eval_n=50, eval_every=400,
           log_every=200, ckpt_every=400)


def collect(env, params, n, seed):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 77)
    kt, kr = jax.random.split(key)
    tasks = jax.vmap(env["sample_task"])(jax.random.split(kt, n))

    def one(task, k):
        h0 = jnp.zeros(hidden_size(params))
        carry0 = (h0, jnp.zeros(OBS_DIM), jnp.zeros(N_ACT), jnp.float32(0.0), k)

        def step(carry, _):
            h, obs, la, lr, k = carry
            k, ka, kenv = jax.random.split(k, 3)
            x = jnp.concatenate([obs, la, jnp.array([lr, 1.0])])
            h, logits = gru_step(params, h, x)
            a = jax.random.categorical(ka, logits)
            _, obs2, r, _, _ = env["step"]((), a, kenv, task)
            return (h, obs2, jax.nn.one_hot(a, N_ACT), r, k), (a, r)

        _, (acts, rews) = jax.lax.scan(step, carry0, None, length=T)
        return acts, rews

    acts, rews = jax.vmap(one)(tasks, jax.random.split(kr, n))
    return np.asarray(acts, np.int8), np.asarray(rews), np.asarray(tasks)


def wanting_metrics(env, params, n=2000, seed=31):
    acts, rews, tasks = collect(env, params, n, seed)
    best = tasks.argmax(1)
    rep = acts[:, 1:] == acts[:, :-1]
    lost = rews[:, :-1] < 0.5
    return dict(
        arms25=float(np.mean([len(set(a[:25])) for a in acts[:1000]])),
        p_rep_loss=float(rep[lost].mean()),
        gate_q4=float(np.mean(acts[:, Q4:] == best[:, None])),
        reward_q4=float(rews[:, Q4:].mean()),
    )


def main():
    rows = []
    for needle in (False, True):
        for fitness in ("late", "q4"):
            tag = f"ES {'needle' if needle else 'uniform'} fitness={fitness}"
            print(f"\n##### {tag}")
            cfg = dict(TOY, fitness=fitness,
                       env_kwargs=dict(needle=needle),
                       out=f"runs/explore_matrix/{int(needle)}_{fitness}")
            theta, unravel = train(cfg)
            env = bandit_env(needle=needle)
            rows.append((tag, wanting_metrics(env, unravel(theta))))

    # fifth specimen: R2 PPO session-1 agent (uniform world, late fitness)
    ppo_ckpt = Path("runs/kaggle_r2/gate_bandit_ppo/ckpt.npz")
    if ppo_ckpt.exists():
        z = np.load(ppo_ckpt, allow_pickle=False)
        _, unr = ravel_pytree(init_ppo_params(jax.random.PRNGKey(0), 128))
        gru = unr(jnp.asarray(z["theta"]))["gru"]
        rows.append(("R2 PPO s1 agent (uniform, late, h=128)",
                     wanting_metrics(bandit_env(), gru)))
    rows.append(("ES fossil h=128 (reference: arms25=2.77, P(rep|L)=0.81)",
                 None))

    print("\n--- wanting-to-explore assay ---")
    print(f"{'condition':50s} {'arms25':>7s} {'P(rep|L)':>9s} {'gate_q4':>8s} {'rew_q4':>7s}")
    lines = []
    for tag, m in rows:
        if m is None:
            line = tag
        else:
            line = (f"{tag:50s} {m['arms25']:7.2f} {m['p_rep_loss']:9.3f} "
                    f"{m['gate_q4']:8.3f} {m['reward_q4']:7.3f}")
        print(line)
        lines.append(line)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "explore_matrix.md").write_text(
        "# Exploration-economics assay (toy ES, 800 gens, h=32)\n\n```\n"
        + "\n".join(lines) + "\n```\n")
    print("\nwrote reports/explore_matrix.md")


if __name__ == "__main__":
    main()
