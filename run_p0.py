#!/usr/bin/env python3
"""Phase 0 entry point. PREREG_P0.md gate config is the default; smoke-scale
runs override via flags.

  python run_p0.py --env bandit --out runs/smoke --gens 200 \
      --hidden 32 --pop 128 --lifetimes 4          # smoke
  python run_p0.py --env bandit --out runs/gate_bandit       # gate config
  python run_p0.py --env catch --out runs/gate_catch --resume runs/gate_catch/ckpt.npz
"""
import argparse

from changeling.ppo import train_ppo
from changeling.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["bandit", "catch"], required=True)
    p.add_argument("--route", choices=["es", "ppo"], default="es")
    p.add_argument("--out", required=True)
    p.add_argument("--gens", type=int, default=3000)
    p.add_argument("--hidden", type=int, default=128)   # PREREG gate value
    p.add_argument("--pop", type=int, default=256)      # PREREG gate value
    p.add_argument("--lifetimes", type=int, default=8)  # PREREG gate value
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--resume", default=None)
    a = p.parse_args()

    config = dict(env=a.env, out=a.out, gens=a.gens, hidden=a.hidden,
                  pop=a.pop, n_lifetimes=a.lifetimes, sigma=a.sigma, lr=a.lr,
                  seed=a.seed, eval_n=a.eval_n, eval_every=a.eval_every,
                  log_every=a.log_every, ckpt_every=a.ckpt_every)
    if a.route == "ppo":
        # R2 defaults (PREREG_P0): same substrate/envs, PPO outer loop
        config.update(updates=a.gens, n_lifetimes=max(a.lifetimes, 64),
                      lr=5e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4,
                      vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5)
        train_ppo(config, resume=a.resume)
    else:
        train(config, resume=a.resume)


if __name__ == "__main__":
    main()
