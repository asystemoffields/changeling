#!/usr/bin/env python3
"""Mixture-curriculum test (assay-informed path to G0-A): does training R2
on a uniform+needle mixture beat uniform-only training, when EVALUATED ON
UNIFORM (the gate task)? Toy scale, both arms matched compute.

Also tests an entropy-pressure variant (higher ent_coef) since session-2
showed PPO settling in policy space (entropy -> 0.01).
"""
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from changeling.envs import bandit_env
from changeling.evaluate import eval_suite
from changeling.ppo import train_ppo

BASE = dict(env="bandit", hidden=32, n_lifetimes=64, lr=5e-4, gamma=0.99,
            lam=0.95, clip=0.2, epochs=4, vf_coef=0.5, ent_coef=0.01,
            max_grad_norm=0.5, seed=0, eval_n=50, eval_every=500,
            log_every=250, ckpt_every=500, updates=1500)

CELLS = [
    ("uniform-trained (control)", dict()),
    ("mixture-trained (mix=0.5)", dict(env_kwargs=dict(mix=0.5))),
    ("uniform + ent_coef=0.03", dict(ent_coef=0.03)),
    ("mixture + ent_coef=0.03", dict(env_kwargs=dict(mix=0.5), ent_coef=0.03)),
]


def main():
    uniform = bandit_env()  # the gate eval task, always
    rows = []
    for tag, over in CELLS:
        print(f"\n##### {tag}")
        cfg = dict(BASE, out=f"runs/mixture_test/{tag.split()[0]}_{over.get('ent_coef', 0.01)}", **over)
        theta, unravel = train_ppo(cfg)
        m = eval_suite(uniform, unravel(theta)["gru"], n=2000, seed=1)
        rows.append((tag, m))

    print("\n--- evaluated on UNIFORM (gate task), q4 ---")
    print(f"{'condition':30s} {'gate_q4':>8s} {'slope':>8s} {'rew_q4':>7s}")
    for tag, m in rows:
        print(f"{tag:30s} {m['gate_q4']:8.3f} {m['slope']:+8.3f} "
              f"{m['reward_q4']:7.3f}")


if __name__ == "__main__":
    main()
