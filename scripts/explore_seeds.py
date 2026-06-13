#!/usr/bin/env python3
"""Multi-seed hardening of the exploration-economics assay (explore_matrix.py is
single-seed). Trains toy ES on {uniform, needle, mix 0.25/0.5/0.75} curricula
across seeds, evaluates EVERY cell on the PURE UNIFORM gate task (the
prereg-relevant held-out readout), and reports mean±std of the wanting metrics.

Question for PREREG_P1: is "the world teaches wanting" (needle/mixture curricula
buy uniform-gate best-arm + exploration) robust to seed, or a single-seed fluke?

  /data/changeling/.venv/bin/python scripts/explore_seeds.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explore_matrix import wanting_metrics, TOY          # noqa: E402
from changeling.envs import bandit_env                   # noqa: E402
from changeling.train import train                       # noqa: E402

SEEDS = [0, 1, 2, 3]
CELLS = [
    ("uniform", dict()),
    ("needle", dict(needle=True)),
    ("mix0.25", dict(mix=0.25)),
    ("mix0.50", dict(mix=0.5)),
    ("mix0.75", dict(mix=0.75)),
]
UNIFORM = bandit_env()  # pure held-out gate task — all cells graded here
METRICS = ["gate_q4", "arms25", "p_rep_loss", "reward_q4"]


def run():
    agg = {}
    for name, ek in CELLS:
        per = []
        for s in SEEDS:
            cfg = dict(TOY, seed=s, fitness="late", env_kwargs=ek,
                       eval_env_kwargs=dict(),  # pure-gate eval (assert_pure_gate)
                       out=f"runs/explore_seeds/{name}_s{s}")
            theta, unravel = train(cfg)
            m = wanting_metrics(UNIFORM, unravel(theta), n=1000, seed=31 + s)
            per.append(m)
            print(f"  [{name} s{s}] gate_q4={m['gate_q4']:.3f} "
                  f"arms25={m['arms25']:.2f} P(rep|L)={m['p_rep_loss']:.3f} "
                  f"rew_q4={m['reward_q4']:.3f}", flush=True)
        agg[name] = per

    lines = [f"{'cell (eval=uniform)':12s} " + " ".join(f"{k:>16s}" for k in METRICS)]
    for name, _ in CELLS:
        per = agg[name]
        cells = []
        for k in METRICS:
            v = np.array([p[k] for p in per])
            cells.append(f"{v.mean():.3f}±{v.std():.3f}")
        lines.append(f"{name:12s} " + " ".join(f"{c:>16s}" for c in cells))
    body = "\n".join(lines)
    print("\n--- multi-seed exploration-economics (mean±std over seeds " +
          f"{SEEDS}) ---\n" + body, flush=True)

    out = Path("reports/explore_seeds.md")
    out.write_text(
        f"# Exploration-economics, multi-seed (toy ES h=32, 800 gens, "
        f"seeds {SEEDS})\n\nAll cells trained on their curriculum, evaluated on "
        f"the PURE UNIFORM gate (n=1000 lifetimes).\n\n```\n" + body + "\n```\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    run()
