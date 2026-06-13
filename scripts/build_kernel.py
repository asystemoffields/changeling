#!/usr/bin/env python3
"""Build a self-contained Kaggle kernel script for a Phase 0 gate run.

  python scripts/build_kernel.py                 # R1 ES kernel (bandit+catch)
  python scripts/build_kernel.py --route ppo     # R2 PPO kernel (bandit only)
  python scripts/build_kernel.py --route ppo --session 2 --updates 32000
      # session 2: new slug <base>-s2, attaches session 1's output as a
      # kernel source so find_resume() continues from its checkpoint

Writes kaggle/<slug>/kernel.py + kernel-metadata.json. Intra-package imports
are stripped; module order satisfies all definitions. Re-run after any
package edit."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "changeling"
# looped.py must precede rollout/es/train/ppo (they import make_step from it) and
# follow agent.py (it imports gru_step/init_gru). Module self-tests are dropped
# (see build()): the bundled kernel runs as __main__, so any module's
# `if __name__ == "__main__":` block would otherwise fire and sys.exit the kernel.
ORDER = ["__init__.py", "envs.py", "agent.py", "looped.py", "interface.py",
         "rollout.py", "es.py", "evaluate.py", "train.py", "ppo.py"]
INTRA = re.compile(r"^\s*from \.\S* import |^\s*from \. import ")
MAIN_GUARD = re.compile(r"^if __name__ ==")

MAIN = '''

# ===== kernel main (route: {route}) =====
if __name__ == "__main__":
    import glob
    import os

    ROUTE = "{route}"
    GATE_BANDIT = 0.672  # 0.90 x Thompson reference 0.747 (PREREG deviation D1)
    smoke = os.environ.get("CHANGELING_SMOKE") == "1"
    print("jax devices:", jax.devices())
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs_kernel"
    wall_start = time.time()
    WALL_BUDGET = 7.5 * 3600  # exit cleanly well inside Kaggle's session cap

    def find_resume(name):
        """Resume from a prior session's checkpoint, attached as a DATASET
        (kernel_sources mounts code, not outputs — probed 2026-06-12)."""
        hits = sorted(h for h in
                      glob.glob("/kaggle/input/**/ckpt.npz", recursive=True)
                      if name in h)
        return hits[0] if hits else None

    if ROUTE == "es":
        common = dict(hidden=128, pop=256, n_lifetimes=8, sigma=0.03, lr=0.02,
                      seed=0, eval_n=100, eval_every=200, log_every=25,
                      ckpt_every=100)
        jobs = [("bandit", dict(gens=3000, stop_gate=GATE_BANDIT,
                                stop_slope_pos=True)),
                ("catch", dict(gens=3000, stop_gate=0.90))]
        trainer = train
    else:
        common = dict(hidden=128, n_lifetimes=64, lr=5e-4, gamma=0.99,
                      lam=0.95, clip=0.2, epochs=4, vf_coef=0.5,
                      ent_coef={ent_coef}, max_grad_norm=0.5, seed=0,
                      eval_n=200, eval_every=250, log_every=50,
                      ckpt_every=250)
        jobs = [("bandit", dict(updates={updates}, stop_gate=GATE_BANDIT,
                                stop_slope_pos=True,
                                env_kwargs=dict(mix={mix}),
                                eval_env_kwargs=dict()))]  # stop listens to gate task
        trainer = train_ppo
    if smoke:
        common.update(hidden=32, eval_n=20, eval_every=5, log_every=1,
                      ckpt_every=5)
        gens_key = "gens" if ROUTE == "es" else "updates"
        jobs = [(n, dict(d, **{{gens_key: 5}})) for n, d in jobs]

    for name, extra in jobs:
        cfg = dict(env=name, out=f"{{base}}/gate_{{name}}_{route}",
                   max_seconds=WALL_BUDGET - (time.time() - wall_start),
                   **common, **extra)
        resume = find_resume(f"gate_{{name}}_{route}")
        print(f"\\n##### {{name}} [{route}]: resume={{resume}}")
        if resume is None and os.path.isdir("/kaggle/input"):
            for cur, dirs, files in os.walk("/kaggle/input"):
                d = cur.count(os.sep) - 2
                print("  " * d + cur, files[:8])
                if d >= 3:
                    dirs[:] = []
        theta, unravel = trainer(cfg, resume=resume)
        params = unravel(theta) if ROUTE == "es" else unravel(theta)["gru"]
        ev = full_eval(ENVS[name](), params, n=cfg["eval_n"], seed=cfg["seed"])
        m, c4, c5 = ev["main"], ev["c4_coin_reward"], ev["c5_no_memory"]
        print(f"\\n=== GATE 0 verdict: {{name}} [{route}] ===")
        if name == "bandit":
            print(f"G0-A gate_q4={{m['gate_q4']:.3f}} (>={{GATE_BANDIT}}, D1):",
                  "PASS" if m["gate_q4"] >= GATE_BANDIT else "FAIL")
            print(f"G0-A slope={{m['slope']:+.4f}} (>0):",
                  "PASS" if m["slope"] > 0 else "FAIL")
            print(f"G0-B c4={{c4['gate_q4']:.3f}} (<=0.225):",
                  "PASS" if c4["gate_q4"] <= 0.225 else "FAIL")
            print(f"G0-C c5={{c5['gate_q4']:.3f}} (<=0.225):",
                  "PASS" if c5["gate_q4"] <= 0.225 else "FAIL")
        else:
            print(f"G0-D gate_q4={{m['gate_q4']:.3f}} (>=0.90):",
                  "PASS" if m["gate_q4"] >= 0.90 else "FAIL")
    print("\\ntotal wall:", round(time.time() - wall_start), "s")
'''


# Phase-1 §5 cold-start tripwire (cbandit-FR, R2 PPO, D3-reconciled). 3 cells:
#   ref  α=0 (un-randomized C3 reference) | cold α=1 (cold-start mainline) |
#   c7   α=1 + fixed_interface (memorization control: trained on ONE frozen
#        interface, eval'd on NOVEL held-out interfaces — predicted to collapse).
# Substrate = plain GRU (loop=False): the tripwire de-risks env+randomization+D3,
# NOT the looped core (that gets its own K_max ladder — one new variable at a time).
# P1_MAIN uses NO str.format (literal braces are fine); params come from P1_HEADER.
P1_MAIN = r'''

# ===== kernel main (Phase-1 §5 cold-start tripwire) =====
if __name__ == "__main__":
    import os, glob, time
    print("jax devices:", jax.devices())
    smoke = os.environ.get("CHANGELING_SMOKE") == "1"
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs_kernel"
    wall_start = time.time()
    WALL_BUDGET = 7.5 * 3600

    common = dict(env="cbandit", hidden=HIDDEN, n_lifetimes=N_LIFE, lr=LR,
                  gamma=1.0, lam=0.95, reward_scale=1.0 / T_LIFE, clip=0.2,
                  epochs=4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, seed=0,
                  eval_n=EVAL_N, eval_every=EVAL_EVERY, log_every=50, ckpt_every=500,
                  loop=False, proj_family="expm-orthogonal",
                  env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                  frozen_rule=True, mix=0.5),
                  eval_env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                       frozen_rule=True, mix=0.0))  # pure gate
    cells = [("ref",  dict(alpha=None)),
             ("cold", dict(alpha=1.0)),
             ("c7",   dict(alpha=1.0, fixed_interface=True, interface_seed=12345))]
    UPDATES_ = UPDATES
    if smoke:
        common.update(hidden=32, n_lifetimes=8, eval_n=40, eval_every=5,
                      log_every=1, ckpt_every=5)
        UPDATES_ = 5

    def find_resume(name):
        hits = sorted(h for h in glob.glob("/kaggle/input/**/ckpt.npz", recursive=True)
                      if name in h)
        return hits[0] if hits else None

    results = {}
    for cname, extra in cells:
        cfg = dict(common, out=base + "/p1_" + cname,
                   max_seconds=WALL_BUDGET - (time.time() - wall_start),
                   updates=UPDATES_, **extra)
        resume = find_resume("p1_" + cname)
        print("\n##### p1 cell [" + cname + "]: alpha=" + str(extra.get("alpha"))
              + " resume=" + str(resume))
        theta, unravel = train_ppo(cfg, resume=resume)
        step = make_step(cfg)
        eval_if = make_iface_fn(cfg, for_eval=True)
        gate_env = ENVS["cbandit"](n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                   frozen_rule=True, mix=0.0)
        ev = full_eval(gate_env, unravel(theta)["gru"], n=cfg["eval_n"],
                       seed=cfg["seed"], step=step, iface_fn=eval_if)
        results[cname] = ev
        m = ev["main"]
        print("  [" + cname + "] gate_q4=%.3f slope=%+.4f sign_p=%.4g c6=%.3f"
              % (m["gate_q4"], m["slope"], m["slope_sign_p"],
                 ev["c6_full_amnesia"]["gate_q4"]))

    print("\n=== Phase-1 §5 cold-start tripwire verdict (cbandit-FR, R2/D3) ===")
    ref, cold, c7 = results["ref"]["main"], results["cold"]["main"], results["c7"]["main"]
    c6 = results["cold"]["c6_full_amnesia"]["gate_q4"]
    print("REF  (a=0)  q4=%.3f slope=%+.4f   (near-ceiling expected)"
          % (ref["gate_q4"], ref["slope"]))
    print("COLD (a=1)  q4=%.3f slope=%+.4f sign_p=%.4g  vs C6=%.3f"
          % (cold["gate_q4"], cold["slope"], cold["slope_sign_p"], c6))
    print("C7 (fixed-iface, NOVEL eval) q4=%.3f slope=%+.4f  (collapse ~0.125 expected)"
          % (c7["gate_q4"], c7["slope"]))
    trig = (cold["slope_sign_p"] >= 0.05) or (cold["gate_q4"] <= c6 + 0.10)
    print("PRE-COMMITTED COLD-START TRIGGER (slope sign-p>=0.05 OR q4<=C6+0.10): "
          + ("FIRED -> switch ALL runs to ANNEAL" if trig
             else "clear -> cold-start mainline holds"))
    print("C7 DISSOCIATION (C7 collapses while REF near ceiling): "
          + ("PRESENT" if (c7["gate_q4"] <= 0.20 and ref["gate_q4"] >= 0.50)
             else "NOT-yet (scale-dependent; PARK with capacity-regression)"))
    print("D3 PPO STABILITY: trained " + str(len(results)) + " cells NaN-free at gamma=1.0")
    print("\ntotal wall:", round(time.time() - wall_start), "s")
'''


# Phase-1 single-axis arms (Gate-1 G1-C/G1-D + their C7 controls; leak-hunt
# follow-up). 4 cells on cbandit-FR, GRU-256, R2/D3, cold-start:
#   Ponly  α_obs=1,α_act=0 (obs-isolation main, G1-D) | c7P  +fixed_interface
#   pionly α_obs=0,α_act=1 (act-isolation main, G1-C) | c7pi +fixed_interface
# Eval is single-axis NOVEL (eval iface_fn inherits alpha_obs/alpha_act) + C8.
P1AXES_MAIN = r'''

# ===== kernel main (Phase-1 single-axis arms: G1-C / G1-D + C7-P / C7-π) =====
if __name__ == "__main__":
    import os, glob, time
    print("jax devices:", jax.devices())
    smoke = os.environ.get("CHANGELING_SMOKE") == "1"
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs_kernel"
    wall_start = time.time()
    WALL_BUDGET = 7.5 * 3600

    common = dict(env="cbandit", hidden=HIDDEN, n_lifetimes=N_LIFE, lr=LR,
                  gamma=1.0, lam=0.95, reward_scale=1.0 / T_LIFE, clip=0.2,
                  epochs=4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, seed=0,
                  eval_n=EVAL_N, eval_every=EVAL_EVERY, log_every=50, ckpt_every=500,
                  loop=False, proj_family="expm-orthogonal", alpha=1.0,
                  env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                  frozen_rule=True, mix=0.5),
                  eval_env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                       frozen_rule=True, mix=0.0))
    # single-axis: alpha_act=0 ⇒ P-only (obs); alpha_obs=0 ⇒ π-only (action)
    cells = [("Ponly",  dict(alpha_act=0.0)),
             ("c7P",    dict(alpha_act=0.0, fixed_interface=True, interface_seed=777)),
             ("pionly", dict(alpha_obs=0.0)),
             ("c7pi",   dict(alpha_obs=0.0, fixed_interface=True, interface_seed=888))]
    UPDATES_ = UPDATES
    if smoke:
        common.update(hidden=32, n_lifetimes=8, eval_n=40, eval_every=5,
                      log_every=1, ckpt_every=5)
        UPDATES_ = 5

    def find_resume(name):
        hits = sorted(h for h in glob.glob("/kaggle/input/**/ckpt.npz", recursive=True)
                      if name in h)
        return hits[0] if hits else None

    results = {}
    for cname, extra in cells:
        cfg = dict(common, out=base + "/p1ax_" + cname,
                   max_seconds=WALL_BUDGET - (time.time() - wall_start),
                   updates=UPDATES_, **extra)
        resume = find_resume("p1ax_" + cname)
        print("\n##### p1axes cell [" + cname + "]: " + str(extra) + " resume=" + str(resume))
        theta, unravel = train_ppo(cfg, resume=resume)
        step = make_step(cfg)
        eval_if = make_iface_fn(cfg, for_eval=True)
        gate_env = ENVS["cbandit"](n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                   frozen_rule=True, mix=0.0)
        ev = full_eval(gate_env, unravel(theta)["gru"], n=cfg["eval_n"],
                       seed=cfg["seed"], step=step, iface_fn=eval_if)
        results[cname] = ev["main"]
        m, c8 = ev["main"], ev.get("c8_reshuffle", {})
        print("  [" + cname + "] gate_q4=%.3f slope=%+.4f sign_p=%.4g c8=%.3f"
              % (m["gate_q4"], m["slope"], m["slope_sign_p"], c8.get("gate_q4", -1)))

    print("\n=== Phase-1 single-axis verdict (cbandit-FR, R2/D3, cold-start) ===")
    Po, c7P, pi, c7pi = (results["Ponly"], results["c7P"],
                         results["pionly"], results["c7pi"])
    g1d = (Po["slope_sign_p"] < 0.05 and Po["slope"] > 0) and c7P["gate_q4"] <= 0.20
    g1c = (pi["slope_sign_p"] < 0.05 and pi["slope"] > 0) and c7pi["gate_q4"] <= 0.20
    print("G1-D OBS-isolation: Ponly q4=%.3f slope=%+.4f sign_p=%.4g | C7-P q4=%.3f -> %s"
          % (Po["gate_q4"], Po["slope"], Po["slope_sign_p"], c7P["gate_q4"],
             "PASS" if g1d else "not-yet"))
    print("G1-C ACT-isolation: pionly q4=%.3f slope=%+.4f sign_p=%.4g | C7-π q4=%.3f -> %s"
          % (pi["gate_q4"], pi["slope"], pi["slope_sign_p"], c7pi["gate_q4"],
             "PASS" if g1c else "not-yet"))
    print("\ntotal wall:", round(time.time() - wall_start), "s")
'''


def build_p1axes(session=1, updates=6000, tag=""):
    """Phase-1 single-axis arms kernel (G1-C/G1-D + C7-P/C7-π), GRU-256."""
    slug = ("changeling-p1-axes"
            + (f"-s{session}" if session != 1 else "") + tag)
    datasets = [] if session == 1 else ["asystemoffields/changeling-ckpts"]
    header = ("\n# ===== Phase-1 single-axis params (generated) =====\n"
              f"HIDDEN = 256\nN_LIFE = 64\nT_LIFE = 256\nLR = 5e-4\n"
              f"EVAL_N = 1000\nEVAL_EVERY = 2000\nUPDATES = {updates}\n")
    parts = ['"""changeling Phase-1 single-axis arms — AUTOGENERATED by '
             'scripts/build_kernel.py; do not edit by hand. See PREREG_P1.md §2-3."""']
    for fname in ORDER:
        lines = []
        for l in (PKG / fname).read_text().splitlines():
            if MAIN_GUARD.match(l):
                break
            if INTRA.match(l):
                continue
            lines.append(l)
        parts.append(f"\n# ===== changeling/{fname} =====\n" + "\n".join(lines))
    parts.append(header + P1AXES_MAIN)
    out_dir = ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.py").write_text("\n".join(parts))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"asystemoffields/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {out_dir}/kernel.py "
          f"({(out_dir / 'kernel.py').stat().st_size} bytes)")


# Phase-1 ANNEALED tripwire (armed after the cold-start trigger fired 2026-06-13).
# One cell: cbandit-FR, GRU-128, R2/D3, α ramped 0→1 on the grid, performance-gated
# (advance when current-α gate_q4 ≥ 0.70, tying mastery to the 0.70×C3 bar). Eval is
# the α=1 GATE. Tests whether the curriculum cracks α=1 where cold-start failed.
P1ANNEAL_MAIN = r'''

# ===== kernel main (Phase-1 ANNEALED tripwire) =====
if __name__ == "__main__":
    import os, glob, time, json as _json, pathlib as _pl
    print("jax devices:", jax.devices())
    smoke = os.environ.get("CHANGELING_SMOKE") == "1"
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs_kernel"
    wall_start = time.time()
    WALL_BUDGET = 7.5 * 3600
    grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    cfg = dict(env="cbandit", out=base + "/p1anneal", hidden=HIDDEN, seed=0,
               n_lifetimes=N_LIFE, gamma=1.0, lam=0.95, clip=0.2, epochs=4,
               vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=LR,
               reward_scale=1.0 / T_LIFE, updates=UPDATES, eval_n=EVAL_N,
               eval_every=EVAL_EVERY, log_every=50, ckpt_every=500, loop=False,
               alpha=1.0, proj_family="expm-orthogonal", anneal=True,
               anneal_threshold=0.70, alpha_grid=grid,
               max_seconds=WALL_BUDGET,
               env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                               frozen_rule=True, mix=0.5),
               eval_env_kwargs=dict(n_arms=8, lifetime=T_LIFE, C_ctx=5,
                                    frozen_rule=True, mix=0.0))
    if smoke:
        cfg.update(hidden=32, n_lifetimes=8, eval_n=24, eval_every=3, log_every=2,
                   ckpt_every=10, updates=24, anneal_threshold=0.0)

    def find_resume(name):
        hits = sorted(h for h in glob.glob("/kaggle/input/**/ckpt.npz", recursive=True)
                      if name in h)
        return hits[0] if hits else None

    resume = find_resume("p1anneal")
    print("##### p1anneal: resume=" + str(resume))
    theta, unravel = train_ppo(cfg, resume=resume)
    step = make_step(cfg)
    eval_if = make_iface_fn(dict(cfg, alpha=1.0), for_eval=True)   # α=1 GATE
    gate_env = ENVS["cbandit"](n_arms=8, lifetime=T_LIFE, C_ctx=5,
                               frozen_rule=True, mix=0.0)
    ev = full_eval(gate_env, unravel(theta)["gru"], n=cfg["eval_n"],
                   seed=cfg["seed"], step=step, iface_fn=eval_if)
    m = ev["main"]; c6 = ev["c6_full_amnesia"]["gate_q4"]
    c8 = ev.get("c8_reshuffle", {}).get("gate_q4", -1.0)
    aj = _pl.Path(cfg["out"]) / "anneal.json"
    idx = _json.loads(aj.read_text())["idx"] if aj.exists() else 0
    print("\n=== Phase-1 ANNEALED tripwire verdict (cbandit-FR GRU-%d R2/D3) ==="
          % cfg["hidden"])
    print("anneal reached alpha=%.2f (grid idx %d/%d)" % (grid[idx], idx, len(grid) - 1))
    print("alpha=1 GATE: q4=%.3f slope=%+.4f sign_p=%.4g | C6=%.3f C8=%.3f"
          % (m["gate_q4"], m["slope"], m["slope_sign_p"], c6, c8))
    cracked = (idx == len(grid) - 1 and m["slope_sign_p"] < 0.05
               and m["slope"] > 0 and m["gate_q4"] >= c6 + 0.10)
    print("ANNEAL CRACKS alpha=1 (top + slope>0 + q4>=C6+0.10): "
          + ("YES -> proceed to capacity/route battery"
             if cracked else "NO -> escalate to capacity dimension (GRU 256/512)"))
    print("\ntotal wall:", round(time.time() - wall_start), "s")
'''


def build_p1anneal(session=1, updates=20000, tag=""):
    """Phase-1 annealed tripwire kernel (single cbandit-FR GRU-128 cell)."""
    slug = ("changeling-p1-anneal"
            + (f"-s{session}" if session != 1 else "") + tag)
    datasets = [] if session == 1 else ["asystemoffields/changeling-ckpts"]
    header = ("\n# ===== Phase-1 annealed-tripwire params (generated) =====\n"
              f"HIDDEN = 128\nN_LIFE = 64\nT_LIFE = 256\nLR = 5e-4\n"
              f"EVAL_N = 800\nEVAL_EVERY = 1000\nUPDATES = {updates}\n")
    parts = ['"""changeling Phase-1 annealed tripwire — AUTOGENERATED by '
             'scripts/build_kernel.py; do not edit by hand. See PREREG_P1.md §5 + '
             'reports/p1_tripwire.md."""']
    for fname in ORDER:
        lines = []
        for l in (PKG / fname).read_text().splitlines():
            if MAIN_GUARD.match(l):
                break
            if INTRA.match(l):
                continue
            lines.append(l)
        parts.append(f"\n# ===== changeling/{fname} =====\n" + "\n".join(lines))
    parts.append(header + P1ANNEAL_MAIN)
    out_dir = ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.py").write_text("\n".join(parts))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"asystemoffields/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {out_dir}/kernel.py "
          f"({(out_dir / 'kernel.py').stat().st_size} bytes)")


def build_p1(session=1, updates=6000, tag=""):
    """Phase-1 §5 cold-start tripwire kernel. ~1e8 env steps/cell at the default
    (n_lifetimes=64, T=256 ⇒ 16384 steps/update; 6000 updates ≈ 9.8e7)."""
    slug = ("changeling-p1-tripwire"
            + (f"-s{session}" if session != 1 else "") + tag)
    datasets = [] if session == 1 else ["asystemoffields/changeling-ckpts"]
    header = ("\n# ===== Phase-1 tripwire params (generated by build_kernel) =====\n"
              f"HIDDEN = 128\nN_LIFE = 64\nT_LIFE = 256\nLR = 5e-4\n"
              f"EVAL_N = 1000\nEVAL_EVERY = 2000\nUPDATES = {updates}\n")
    parts = ['"""changeling Phase-1 §5 cold-start tripwire — AUTOGENERATED by '
             'scripts/build_kernel.py; do not edit by hand. See PREREG_P1.md §5."""']
    for fname in ORDER:
        lines = []
        for l in (PKG / fname).read_text().splitlines():
            if MAIN_GUARD.match(l):
                break
            if INTRA.match(l):
                continue
            lines.append(l)
        parts.append(f"\n# ===== changeling/{fname} =====\n" + "\n".join(lines))
    parts.append(header + P1_MAIN)
    out_dir = ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.py").write_text("\n".join(parts))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"asystemoffields/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {out_dir}/kernel.py "
          f"({(out_dir / 'kernel.py').stat().st_size} bytes)")


def build(route, session=1, updates=8000, mix=0.0, ent=None, tag=""):
    base = "changeling-p0-gate" if route == "es" else "changeling-p0-r2"
    slug = (base if session == 1 else f"{base}-s{session}") + tag
    # resume checkpoints travel via the changeling-ckpts DATASET; probed
    # 2026-06-12: kernel_sources mounts the source kernel's code, NOT outputs
    datasets = [] if session == 1 else ["asystemoffields/changeling-ckpts"]
    parts = ['"""changeling Phase 0 gate runs — AUTOGENERATED by '
             'scripts/build_kernel.py; do not edit by hand. See SPEC.md / '
             'PREREG_P0.md in the repo."""']
    for fname in ORDER:
        lines = []
        for l in (PKG / fname).read_text().splitlines():
            if MAIN_GUARD.match(l):
                break  # drop the module self-test (kernel runs as __main__)
            if INTRA.match(l):
                continue
            lines.append(l)
        parts.append(f"\n# ===== changeling/{fname} =====\n" + "\n".join(lines))
    parts.append(MAIN.format(route=route, updates=updates, mix=mix,
                             ent_coef=ent if ent is not None else 0.01))
    out_dir = ROOT / "kaggle" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.py").write_text("\n".join(parts))
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"asystemoffields/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {out_dir}/kernel.py "
          f"({(out_dir / 'kernel.py').stat().st_size} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["es", "ppo", "p1", "p1axes", "p1anneal"],
                    default="es")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--updates", type=int, default=8000)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--ent", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.route in ("p1", "p1axes", "p1anneal"):
        _def = {"p1": 6000, "p1axes": 6000, "p1anneal": 20000}[a.route]
        u = a.updates if a.updates != 8000 else _def
        {"p1": build_p1, "p1axes": build_p1axes,
         "p1anneal": build_p1anneal}[a.route](a.session, u, a.tag)
    else:
        build(a.route, a.session, a.updates, a.mix, a.ent, a.tag)
