#!/usr/bin/env python3
"""Crystallization pilot (Phase 5 preview, see SPEC 1c/Phase 4).

Dissects the plateaued ES bandit agent (Gate 0 session 1, gate_q4~0.25):
  1. collect traces from the fossil
  2. behavioral forensics (what IS the strategy?)
  3. fit an MDL ladder of rule families to the traces (max likelihood)
  4. play every fitted crystal on fresh tasks; find the MDL knee

Two clearly-distinct columns: fit-to-fossil (extracted) vs tuned-for-reward
(the family's ceiling, NOT an extraction).

  .venv/bin/python scripts/crystallize_bandit.py runs/kaggle_out/gate_bandit/ckpt.npz
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from changeling import N_ACT, OBS_DIM
from changeling.agent import gru_step, hidden_size, init_gru
from changeling.es import adam_init, adam_ascend
from changeling.envs import bandit_env

T, K = 200, 8
Q4 = T - T // 4
ENV = bandit_env(n_arms=K, lifetime=T)


# ---------- fossil ----------

def load_fossil(path, hidden=128):
    """Return the GRU params of an ES OR PPO checkpoint (auto-detected from the
    stored config — PPO carries gamma/lam and a value head, ES does not)."""
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["config"])) if "config" in z.files else {}
    hidden = cfg.get("hidden", hidden)
    if "gamma" in cfg:  # PPO: deploy artifact is params["gru"]
        from changeling.ppo import init_ppo_params
        _, unravel = ravel_pytree(init_ppo_params(jax.random.PRNGKey(0), hidden))
        return unravel(jnp.asarray(z["theta"]))["gru"]
    _, unravel = ravel_pytree(init_gru(jax.random.PRNGKey(0), hidden=hidden))
    return unravel(jnp.asarray(z["theta"]))


def collect(params, n, seed):
    """Play n lifetimes; return acts (n,T) int8, rews (n,T), tasks (n,K)."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 77)
    kt, kr = jax.random.split(key)
    tasks = jax.vmap(ENV["sample_task"])(jax.random.split(kt, n))

    def one(task, k):
        h0 = jnp.zeros(hidden_size(params))
        carry0 = (h0, jnp.zeros(OBS_DIM), jnp.zeros(N_ACT), jnp.float32(0.0), k)

        def step(carry, _):
            h, obs, la, lr, k = carry
            k, ka, kenv = jax.random.split(k, 3)
            x = jnp.concatenate([obs, la, jnp.array([lr, 1.0])])
            h, logits = gru_step(params, h, x)
            a = jax.random.categorical(ka, logits)
            _, obs2, r, _, _ = ENV["step"]((), a, kenv, task)
            return (h, obs2, jax.nn.one_hot(a, N_ACT), r, k), (a, r)

        _, (acts, rews) = jax.lax.scan(step, carry0, None, length=T)
        return acts, rews

    acts, rews = jax.vmap(one)(tasks, jax.random.split(kr, n))
    return np.asarray(acts, np.int8), np.asarray(rews), np.asarray(tasks)


# ---------- forensics ----------

def forensics(acts, rews, tasks):
    n = acts.shape[0]
    best = tasks.argmax(1)
    rep = acts[:, 1:] == acts[:, :-1]
    won = rews[:, :-1] > 0.5
    print("\n--- forensics ---")
    for lo, hi, tag in [(0, 50, "q1"), (Q4, T, "q4")]:
        sl = slice(max(lo, 0), hi - 1)
        print(f"{tag}: best-arm={np.mean(acts[:, lo:hi] == best[:, None]):.3f} "
              f"reward={rews[:, lo:hi].mean():.3f} "
              f"P(repeat|win)={rep[:, sl][won[:, sl]].mean():.3f} "
              f"P(repeat|loss)={rep[:, sl][~won[:, sl]].mean():.3f}")
    print(f"distinct arms tried in first 25 steps: "
          f"{np.mean([len(set(a[:25])) for a in acts[:500]]):.2f} / {K}")
    # value tracking: P(action == argmax of running empirical mean)
    track = np.zeros(2)
    cnt = np.full((n, K), 1e-6)
    tot = np.zeros((n, K))
    idx = np.arange(n)
    for t in range(1, T):
        track += [np.mean(acts[:, t] == (tot / cnt).argmax(1)), 1] if t >= Q4 else 0
        cnt[idx, acts[:, t]] += 1
        tot[idx, acts[:, t]] += rews[:, t]
    print(f"q4 P(action == argmax running mean): {track[0] / track[1]:.3f}")


# ---------- rule families (numpy, vectorized over lifetimes) ----------

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def q_probs_stream(acts, rews, alpha, beta, stick, q0):
    """Per-step action probabilities of a softmax-Q+perseveration rule run
    along the fossil's own (a, r) stream. alpha=None -> empirical mean."""
    n = acts.shape[0]
    Q = np.full((n, K), q0)
    cnt = np.zeros((n, K))
    last = np.zeros((n, K))
    idx = np.arange(n)
    nll, m = 0.0, 0
    for t in range(T):
        p = softmax(beta * Q + stick * last)
        nll -= np.log(p[idx, acts[:, t]] + 1e-12).sum()
        m += n
        a, r = acts[:, t], rews[:, t]
        cnt[idx, a] += 1
        lr = alpha if alpha is not None else 1.0 / cnt[idx, a]
        Q[idx, a] += lr * (r - Q[idx, a])
        last[:] = 0.0
        last[idx, a] = 1.0
    return nll / m


def play_q(tasks, alpha, beta, stick, q0, seed):
    rng = np.random.default_rng(seed)
    n = tasks.shape[0]
    best = tasks.argmax(1)
    Q = np.full((n, K), q0)
    cnt = np.zeros((n, K))
    last = np.zeros((n, K))
    idx = np.arange(n)
    hit = rew = 0.0
    for t in range(T):
        p = softmax(beta * Q + stick * last)
        a = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        r = (rng.random(n) < tasks[idx, a]).astype(float)
        if t >= Q4:
            hit += (a == best).mean()
            rew += r.mean()
        cnt[idx, a] += 1
        lr = alpha if alpha is not None else 1.0 / cnt[idx, a]
        Q[idx, a] += lr * (r - Q[idx, a])
        last[:] = 0.0
        last[idx, a] = 1.0
    return hit / (T - Q4), rew / (T - Q4)


def play_wsls(tasks, p_stay_win, p_stay_loss, seed):
    rng = np.random.default_rng(seed)
    n = tasks.shape[0]
    best = tasks.argmax(1)
    a = rng.integers(0, K, n)
    idx = np.arange(n)
    hit = rew = 0.0
    r = np.zeros(n)
    for t in range(T):
        if t:
            stay_p = np.where(r > 0.5, p_stay_win, p_stay_loss)
            move = rng.random(n) >= stay_p
            a = np.where(move, rng.integers(0, K, n), a)
        r = (rng.random(n) < tasks[idx, a]).astype(float)
        if t >= Q4:
            hit += (a == best).mean()
            rew += r.mean()
    return hit / (T - Q4), rew / (T - Q4)


def fit_wsls(acts, rews):
    rep = acts[:, 1:] == acts[:, :-1]
    won = rews[:, :-1] > 0.5
    return float(rep[won].mean()), float(rep[~won].mean())


def grid_fit_q(acts, rews, eps_greedy=False):
    """Return (best_params, nll). eps_greedy=True restricts to the
    high-beta + uniform-mix corner via stick=0, beta large, alpha grid."""
    grids = dict(
        alpha=[None, 0.05, 0.1, 0.2, 0.35, 0.5],
        beta=[2, 4, 8, 16, 32] if not eps_greedy else [50],
        stick=[0, 0.5, 1, 2] if not eps_greedy else [0],
        q0=[0.0, 0.5, 1.0],
    )
    best = (None, np.inf)
    for al in grids["alpha"]:
        for be in grids["beta"]:
            for st in grids["stick"]:
                for q0 in grids["q0"]:
                    nll = q_probs_stream(acts, rews, al, be, st, q0)
                    if nll < best[1]:
                        best = (dict(alpha=al, beta=be, stick=st, q0=q0), nll)
    return best


# ---------- distillation ----------

def distill(acts, rews, hidden, iters=400, lr=3e-3, seed=5):
    """Behavior-clone the fossil into a tiny GRU; return params."""
    n = acts.shape[0]
    X = np.zeros((n, T, OBS_DIM + N_ACT + 2), np.float32)
    X[:, :, -1] = 1.0  # boundary bit (bandit: every step)
    X[:, 1:, OBS_DIM:OBS_DIM + N_ACT] = np.eye(N_ACT, dtype=np.float32)[acts[:, :-1]]
    X[:, 1:, OBS_DIM + N_ACT] = rews[:, :-1]
    X, Y = jnp.asarray(X), jnp.asarray(acts.astype(np.int32))

    params = init_gru(jax.random.PRNGKey(seed), hidden=hidden)
    theta, unravel = ravel_pytree(params)
    adam = adam_init(theta.shape[0])

    def loss(th, xb, yb):
        p = unravel(th)

        def fwd(xs):
            def step(h, x):
                h, logits = gru_step(p, h, x)
                return h, logits
            _, logits = jax.lax.scan(step, jnp.zeros(hidden), xs)
            return logits

        logits = jax.vmap(fwd)(xb)
        lp = jax.nn.log_softmax(logits)
        return -jnp.take_along_axis(lp, yb[..., None], -1).mean()

    grad_fn = jax.jit(jax.value_and_grad(loss))
    key = jax.random.PRNGKey(seed + 1)
    for i in range(iters):
        key, kb = jax.random.split(key)
        ix = jax.random.choice(kb, n, (128,), replace=False)
        l, g = grad_fn(theta, X[ix], Y[ix])
        theta, adam = adam_ascend(theta, -g, adam, lr)
        if (i + 1) % 100 == 0:
            print(f"  distill h={hidden} iter {i + 1} ce={float(l):.4f}")
    return unravel(theta), int(theta.shape[0])


# ---------- main ----------

def main(ckpt):
    fossil = load_fossil(ckpt)
    print("collecting fossil traces...")
    acts, rews, tasks = collect(fossil, n=2000, seed=123)
    forensics(acts, rews, tasks)

    # fresh eval tasks for *playing* every candidate
    eval_tasks = np.asarray(jax.vmap(ENV["sample_task"])(
        jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), 99), 4000)))
    rows = []

    e_acts, e_rews, e_tasks = collect(fossil, n=4000, seed=999)
    e_best = e_tasks.argmax(1)
    rows.append(("fossil GRU-128 (organism)",
                 int(ravel_pytree(fossil)[0].shape[0]), None,
                 float(np.mean(e_acts[:, Q4:] == e_best[:, None])),
                 float(e_rews[:, Q4:].mean())))

    p1, p0 = fit_wsls(acts, rews)
    hit, rew = play_wsls(eval_tasks, p1, p0, 7)
    rows.append((f"C1 WSLS fit (stay|w={p1:.2f}, stay|l={p0:.2f})", 2, None, hit, rew))

    prm, nll = grid_fit_q(acts, rews, eps_greedy=True)
    hit, rew = play_q(eval_tasks, prm["alpha"], prm["beta"], prm["stick"], prm["q0"], 8)
    rows.append((f"C2 eps-greedy-ish fit {prm}", 3, nll, hit, rew))

    prm, nll = grid_fit_q(acts, rews)
    hit, rew = play_q(eval_tasks, prm["alpha"], prm["beta"], prm["stick"], prm["q0"], 9)
    rows.append((f"C3 softmax-Q+stick fit {prm}", 4, nll, hit, rew))

    # family ceiling — tuned for reward, NOT an extraction
    best_t = (None, -1, -1)
    for al in [None, 0.05, 0.1, 0.2]:
        for be in [4, 8, 16, 32]:
            for q0 in [0.5, 1.0]:
                h, r = play_q(eval_tasks[:2000], al, be, 0, q0, 11)
                if r > best_t[2]:
                    best_t = (dict(alpha=al, beta=be, q0=q0), h, r)
    rows.append((f"C3* tuned-for-reward (ceiling) {best_t[0]}", 4, None,
                 best_t[1], best_t[2]))

    for h in (4, 8):
        small, npar = distill(acts, rews, hidden=h)
        d_acts, d_rews, d_tasks = collect(small, n=4000, seed=999)
        rows.append((f"C4 distilled GRU-{h}", npar, None,
                     float(np.mean(d_acts[:, Q4:] == d_tasks.argmax(1)[:, None])),
                     float(d_rews[:, Q4:].mean())))

    rows.append(("Thompson reference", None, None, 0.747, None))

    print("\n--- MDL ladder (played on fresh tasks; q4 = final quarter) ---")
    print(f"{'family':55s} {'params':>7s} {'NLL/step':>9s} {'best-arm':>9s} {'reward':>7s}")
    lines = []
    for name, npar, nll, hit, rew in rows:
        line = (f"{name:55s} {str(npar):>7s} "
                f"{f'{nll:.4f}' if nll else '-':>9s} {hit:9.3f} "
                f"{f'{rew:.3f}' if rew is not None else '-':>7s}")
        print(line)
        lines.append(line)
    out = Path("runs/crystal_" + Path(ckpt).parent.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(
        f"# Crystallization — {ckpt}\n\n```\n"
        + "\n".join(lines) + "\n```\n")
    print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "runs/kaggle_out/gate_bandit/ckpt.npz")
