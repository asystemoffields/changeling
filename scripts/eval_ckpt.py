#!/usr/bin/env python3
"""Score a Phase 0 checkpoint against its Gate 0 criterion, at high n and
across seeds. Auto-detects ES vs PPO and hidden size from the stored config.

The GATE eval task is always the pure held-out distribution (bandit: uniform
U(0,1); catch: catch) regardless of the training distribution — mixture
curricula (D2) train on mix>0 but are graded on pure uniform.

  .venv/bin/python scripts/eval_ckpt.py runs/.../ckpt.npz
  .venv/bin/python scripts/eval_ckpt.py runs/.../ckpt.npz --n 2000 --seeds 1,2,3
"""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from changeling.agent import init_gru
from changeling.envs import bandit_env, catch_env
from changeling.evaluate import full_eval
from changeling.ppo import init_ppo_params

# D1: 0.90 x Thompson reference 0.747. Catch: locked G0-D.
BARS = {"bandit": 0.672, "catch": 0.90}


def load(ckpt):
    z = np.load(ckpt, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    hidden = cfg["hidden"]
    is_ppo = "gamma" in cfg  # PPO config carries gamma/lam; ES does not
    if is_ppo:
        _, unravel = ravel_pytree(init_ppo_params(jax.random.PRNGKey(0), hidden))
        params = unravel(jnp.asarray(z["theta"]))["gru"]
    else:
        _, unravel = ravel_pytree(init_gru(jax.random.PRNGKey(0), hidden=hidden))
        params = unravel(jnp.asarray(z["theta"]))
    return params, cfg, ("ppo" if is_ppo else "es")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()

    params, cfg, route = load(a.ckpt)
    envname = cfg["env"]
    env = bandit_env() if envname == "bandit" else catch_env()  # pure gate task
    bar = BARS[envname]
    print(f"ckpt={a.ckpt}\nroute={route} env={envname} hidden={cfg['hidden']} "
          f"step={cfg.get('gens', cfg.get('updates', '?'))}  n={a.n}  bar={bar}")

    gates, slopes = [], []
    print(f"\n{'seed':>4s} {'gate_q4':>8s} {'slope':>8s} {'sign_p':>10s} "
          f"{'c4':>6s} {'c5':>6s} {'c6':>6s}")
    for s in [int(x) for x in a.seeds.split(",")]:
        ev = full_eval(env, params, n=a.n, seed=s)
        m = ev["main"]
        gates.append(m["gate_q4"])
        slopes.append(m["slope"])
        print(f"{s:>4d} {m['gate_q4']:8.3f} {m['slope']:+8.3f} "
              f"{m['slope_sign_p']:10.2e} "
              f"{ev['c4_coin_reward']['gate_q4']:6.3f} "
              f"{ev['c5_no_memory']['gate_q4']:6.3f} "
              f"{ev['c6_full_amnesia']['gate_q4']:6.3f}")

    gmean = float(np.mean(gates))
    print(f"\nmean gate_q4 = {gmean:.4f}  (bar {bar})  "
          f"slope>0 all seeds: {all(x > 0 for x in slopes)}")
    verdict = "PASS" if gmean >= bar and all(x > 0 for x in slopes) else "FAIL"
    print(f"G0 verdict: {verdict}")
    return gmean, verdict


if __name__ == "__main__":
    main()
