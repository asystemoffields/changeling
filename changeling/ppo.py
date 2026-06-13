"""Route R2: PPO-RL^2 (Duan et al. 2016 protocol, clipped PPO objective).

Whole lifetimes are the trajectories; BPTT through the full lifetime; value
function does NOT reset at episode boundaries (the objective is lifetime
return). Training rewards carry the same late-weight w_t as R1 fitness, but the
selection pressures are NOT identical: R1 ranks the undiscounted weighted-average
fitness sum(w_t·r_t)/sum(w_t), whereas here w_t enters the per-step reward and is
then reshaped by the γ-discounted GAE — the effective per-step weight is γ^t·w_t,
which for γ=0.99 over T=200 tilts EARLY, the reverse of R1's late tilt. This is a
DECLARED confounder of the R1-vs-R2 slope comparison (PREREG_P0 D3), to be
controlled in the Phase-1 route-scaling protocol; it does not affect any
single-route eval (eval is untouched and uses raw rewards).

The deploy artifact is params["gru"] — the same pure S1 substrate as R1.
The value head exists only at training time (a connector, in SPEC terms).
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import IN_DIM, N_ACT
from .agent import gru_step, init_gru, hidden_size
from .looped import init_looped, make_step, kl_to_geometric
from .interface import IFACE_FOLD, make_iface_fn
from .es import adam_init, adam_ascend
from .envs import ENVS
from .evaluate import full_eval
from .train import save_ckpt, load_ckpt, _flat, _log, assert_resume_cfg, assert_pure_gate


def init_ppo_params(key, hidden=128, cfg=None):
    """Legacy: a plain GRU core + a separate top-level value head (a training-only
    connector). Looped (cfg['loop']): the looped core, whose halt+value heads live
    INSIDE params['gru'] — commit_v IS the state-value, so no top-level Wv/bv."""
    cfg = cfg or {}
    if cfg.get("loop", False):
        return {"gru": init_looped(key, hidden)}
    kg, kv = jax.random.split(key)
    return {
        "gru": init_gru(kg, hidden),
        "Wv": jax.random.normal(kv, (hidden, 1)) / jnp.sqrt(hidden),
        "bv": jnp.zeros(1),
    }


def collect(env, params, task, key, reward_scale, step, loop, iface=None):
    """One lifetime under the current policy. Returns everything the loss
    needs to recompute the forward pass exactly (teacher forcing on xs).
    Rewards come back late-weighted (PREREG_P0 selection pressure) and
    scaled so discounted returns are O(1) — otherwise the value-loss
    gradient through the shared trunk drowns the policy gradient.

    `step` is a B1 step_fn (looped.make_step). Legacy: sample from raw logits,
    value from the top-level head — byte-identical to the pre-B1 collect. Looped:
    sample from the marginal commit dist; the state-value IS commit_v.

    `iface` (B2) is the per-lifetime (P, perm). xs store the PROJECTED obs and the
    RAW action, so the loss replay (teacher-forced on xs) stays the exact forward
    pass — the collect==forward lynchpin is preserved under randomization."""
    T = env["T"]
    if iface is not None:
        P, perm = iface
    key, k0 = jax.random.split(key)
    state0, obs0 = env["reset"](k0, task)
    h0 = jnp.zeros(hidden_size(params["gru"]))
    carry0 = (h0, state0, obs0, jnp.zeros(N_ACT), jnp.float32(0.0),
              jnp.float32(1.0), key)

    def step_fn(carry, w_t):
        h, state, obs, last_a, last_r, boundary, key = carry
        key, ka, kr, kres = jax.random.split(key, 4)
        obs_in = P @ obs if iface is not None else obs
        x = jnp.concatenate([obs_in, last_a, jnp.array([last_r, boundary])])
        h, sample_score, logp_all, value, aux = step(params["gru"], h, x)
        a = jax.random.categorical(ka, sample_score)
        a_env = perm[a] if iface is not None else a
        v = value if loop else (h @ params["Wv"] + params["bv"])[0]
        state, obs, r, done, _ = env["step"](state, a_env, kr, task)
        rs_state, rs_obs = env["reset"](kres, task)
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, new, old), rs_state, state)
        obs = jnp.where(done, rs_obs, obs)
        carry = (h, state, obs, jax.nn.one_hot(a, N_ACT), r,
                 done.astype(jnp.float32), key)
        out = (x, a, logp_all[a], v, r * w_t * reward_scale)
        return carry, out

    w = jnp.linspace(0.5, 1.5, T)
    _, (xs, acts, logps, vals, rews) = jax.lax.scan(step_fn, carry0, w)
    return xs, acts, logps, vals, rews


def gae(rews, vals, gamma, lam):
    """Advantages over the whole lifetime; terminal value 0 (lifetime over)."""
    vals_next = jnp.concatenate([vals[1:], jnp.zeros(1)])
    deltas = rews + gamma * vals_next - vals

    def back(carry, delta):
        carry = delta + gamma * lam * carry
        return carry, carry

    _, adv = jax.lax.scan(back, jnp.float32(0.0), deltas, reverse=True)
    return adv, adv + vals  # advantages, returns


def make_update_step(env, unravel, cfg):
    n_life = cfg["n_lifetimes"]
    gamma, lam = cfg["gamma"], cfg["lam"]
    clip, vf_c, ent_c = cfg["clip"], cfg["vf_coef"], cfg["ent_coef"]

    # B1 looped substrate (trace-time branch: loop=False emits the exact legacy
    # graph). When loop=True the policy is the marginal commit mixture, the value
    # is per-turn v_k reconstructed (PonderNet L_rec), and a KL-to-geometric prior
    # (weight kl_coef) regularizes the halting distribution toward E[K]=1/lam_p.
    loop = cfg.get("loop", False)
    step = make_step(cfg)
    k_max, k_min = int(cfg.get("k_max", 1)), int(cfg.get("k_min", 2))
    lam_p, kl_c = float(cfg.get("lam_p", 0.2)), float(cfg.get("kl_coef", 0.0))
    iface_fn = make_iface_fn(cfg, for_eval=False)   # B2 train-interface sampler

    rew_scale = cfg.get("reward_scale", 1.0 - cfg["gamma"])
    # D3 landmine: at gamma=1.0 (route-reconciled config) the default 1-gamma=0
    # would zero every training reward. Require an explicit positive scale.
    assert rew_scale > 0, (
        f"reward_scale={rew_scale} (<=0) zeros every training reward; with "
        f"gamma={cfg['gamma']} the default 1-gamma is non-positive — pass "
        f"reward_scale explicitly (D3 reconciliation uses 1/T at gamma=1.0).")

    def batch_collect(theta, key):
        params = unravel(theta)
        kt, kr = jax.random.split(key)
        tasks = jax.vmap(env["sample_task"])(jax.random.split(kt, n_life))
        keys = jax.random.split(kr, n_life)
        if iface_fn is None:
            xs, acts, logps, vals, rews = jax.vmap(
                lambda t, k: collect(env, params, t, k, rew_scale, step, loop))(
                    tasks, keys)
        else:
            # interface keys from a disjoint fold (does NOT consume kt/kr above)
            ik = jax.random.split(jax.random.fold_in(key, IFACE_FOLD), n_life)
            ifaces = jax.vmap(iface_fn)(ik)
            xs, acts, logps, vals, rews = jax.vmap(
                lambda t, k, i: collect(env, params, t, k, rew_scale, step, loop,
                                        iface=i))(tasks, keys, ifaces)
        adv, rets = jax.vmap(lambda r, v: gae(r, v, gamma, lam))(rews, vals)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return xs, acts, logps, adv, rets

    def loss_fn(theta, batch):
        params = unravel(theta)
        xs, acts, logps_old, adv, rets = batch
        h0 = jnp.zeros(hidden_size(params["gru"]))

        if loop:
            def forward(xs_one):
                def st(h, x):
                    h, _ss, logp_all, value, aux = step(params["gru"], h, x)
                    return h, (logp_all, value, aux["p"], aux["v_k"], aux["E_K"])
                _, outs = jax.lax.scan(st, h0, xs_one)
                return outs  # (logp_all, v, p, v_k, E_K)

            logp_all, v, p, v_k, e_k = jax.vmap(forward)(xs)
            #   logp_all (B,T,A)  v (B,T)  p (B,T,K)  v_k (B,T,K)  e_k (B,T)
            logp = jnp.take_along_axis(logp_all, acts[..., None], -1)[..., 0]
            ratio = jnp.exp(logp - logps_old)
            pg = jnp.minimum(ratio * adv,
                             jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
            # per-turn value reconstruction, halting-weighted (PonderNet L_rec):
            # each micro-turn's value head is regressed to the lifetime return,
            # weighted by that turn's commit mass p_k.
            vloss = (p * (v_k - rets[..., None]) ** 2).sum(-1).mean()
            ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
            kl_fn = lambda pp: kl_to_geometric(pp, lam_p, k_max, k_min)
            klr = jax.vmap(jax.vmap(kl_fn))(p).mean()
            loss = -(pg - vf_c * vloss + ent_c * ent) + kl_c * klr
            return loss, dict(pg=pg, v=vloss, ent=ent, kl=klr, e_k=e_k.mean())

        def forward(xs_one):
            def st(h, x):
                h, _ss, logp_all, _value, _aux = step(params["gru"], h, x)
                v = (h @ params["Wv"] + params["bv"])[0]
                return h, (logp_all, v)
            _, (logp_all, v) = jax.lax.scan(st, h0, xs_one)
            return logp_all, v

        logp_all, v = jax.vmap(forward)(xs)  # (B,T,A), (B,T)
        logp = jnp.take_along_axis(logp_all, acts[..., None], -1)[..., 0]
        ratio = jnp.exp(logp - logps_old)
        pg = jnp.minimum(ratio * adv,
                         jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
        vloss = 0.5 * ((v - rets) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return -(pg - vf_c * vloss + ent_c * ent), dict(pg=pg, v=vloss, ent=ent)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def epoch_step(theta, adam_state, batch):
        (loss, aux), grad = grad_fn(theta, batch)
        gnorm = jnp.sqrt(jnp.sum(grad ** 2))
        grad = grad * jnp.minimum(1.0, cfg["max_grad_norm"] / (gnorm + 1e-8))
        theta, adam_state = adam_ascend(theta, -grad, adam_state, cfg["lr"])
        return theta, adam_state, loss, dict(aux, gnorm=gnorm)

    return jax.jit(batch_collect), jax.jit(epoch_step)


def train_ppo(config, resume=None):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    env = ENVS[config["env"]](**config.get("env_kwargs", {}))
    # eval (and therefore stop_gate) listens to the GATE task, which may
    # differ from the training distribution (e.g. mixture curricula)
    eval_env = ENVS[config["env"]](**config.get("eval_env_kwargs",
                                                config.get("env_kwargs", {})))
    assert_pure_gate(config, eval_env)
    print(f"R2 grades gate on eval_env={eval_env['name']} "
          f"kwargs={config.get('eval_env_kwargs', config.get('env_kwargs', {}))}")

    step = make_step(config)
    eval_iface_fn = make_iface_fn(config, for_eval=True)   # held-out eval interfaces

    if resume:
        theta, adam_state, key, start_up, saved_cfg, best_gate = load_ckpt(resume)
        assert_resume_cfg(saved_cfg, config)
        _, unravel = ravel_pytree(
            init_ppo_params(jax.random.PRNGKey(0), config["hidden"], config))
        print(f"resumed update={start_up} from {resume} (best_gate={best_gate:.3f})")
    else:
        key = jax.random.PRNGKey(config["seed"])
        key, ki = jax.random.split(key)
        theta, unravel = ravel_pytree(init_ppo_params(ki, config["hidden"], config))
        adam_state = adam_init(theta.shape[0])
        start_up = 0
        best_gate = -1.0

    # Anneal (PREREG §5 fallback, armed when the cold-start trigger fired): ramp the
    # interface strength α through a grid, performance-gated, rebuilding the jitted
    # update step on each advance (re-jit is paid ≤len(grid) times — negligible). The
    # GATE metric stays α=1 (config['alpha']=1.0 ⇒ eval_iface_fn); advancement is gated
    # on performance at the CURRENT grid α. Index persists across resume via
    # out/anneal.json (leaves save_ckpt/load_ckpt untouched).
    anneal = config.get("anneal", False)
    a_grid = config.get("alpha_grid", [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0])
    a_thr = float(config.get("anneal_threshold", 0.7))
    aidx = 0
    if anneal and resume and (out / "anneal.json").exists():
        aidx = int(json.loads((out / "anneal.json").read_text())["idx"])

    def _build(a):
        c = dict(config, alpha=float(a))
        bc, es = make_update_step(env, unravel, c)
        return bc, es, make_iface_fn(c, for_eval=True)

    eval_if_cur = None
    if anneal:
        batch_collect, epoch_step, eval_if_cur = _build(a_grid[aidx])
        print(f"R2 PPO ANNEAL env={env['name']} grid={a_grid} start_idx={aidx} "
              f"thr={a_thr} dim={theta.shape[0]} B={config['n_lifetimes']} T={env['T']}")
    else:
        batch_collect, epoch_step = make_update_step(env, unravel, config)
        print(f"R2 PPO env={env['name']} dim={theta.shape[0]} "
              f"B={config['n_lifetimes']} T={env['T']}")

    max_seconds = config.get("max_seconds")
    stop_gate = config.get("stop_gate")
    stop_slope_pos = config.get("stop_slope_pos", False)

    t0 = time.time()
    for up in range(start_up, config["updates"]):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, up, config,
                      best_gate=best_gate)
            print(f"wall-clock budget reached at update {up}; exiting")
            break
        key, kc = jax.random.split(key)
        batch = batch_collect(theta, kc)
        for _ in range(config["epochs"]):
            theta, adam_state, loss, aux = epoch_step(theta, adam_state, batch)
        if (up + 1) % config["log_every"] == 0:
            row = dict(update=up, sec=round(time.time() - t0, 1),
                       loss=float(loss), pg=float(aux["pg"]),
                       vloss=float(aux["v"]), ent=float(aux["ent"]),
                       gnorm=float(aux["gnorm"]))
            if "kl" in aux:  # looped substrate diagnostics
                row["kl"] = float(aux["kl"])
                row["e_k"] = float(aux["e_k"])
            _log(out, row)
            print(row)
        if (up + 1) % config["eval_every"] == 0 or up + 1 == config["updates"]:
            ev = full_eval(eval_env, unravel(theta)["gru"], n=config["eval_n"],
                           seed=config["seed"], step=step, iface_fn=eval_iface_fn)
            _log(out, dict(update=up, **_flat(ev)))
            print(f"  eval u{up}: main={ev['main']} "
                  f"c4={ev['c4_coin_reward']['gate_q4']:.3f} "
                  f"c5={ev['c5_no_memory']['gate_q4']:.3f}")
            # ANNEAL: advance α when performance at the CURRENT grid level clears the
            # threshold (eval at current α, not the α=1 gate above). Rebuild on advance.
            if anneal and aidx < len(a_grid) - 1:
                ev_cur = full_eval(eval_env, unravel(theta)["gru"], n=config["eval_n"],
                                   seed=config["seed"], step=step, iface_fn=eval_if_cur)
                cur_q4 = ev_cur["main"]["gate_q4"]
                _log(out, dict(update=up, anneal_alpha=a_grid[aidx], anneal_cur_q4=cur_q4))
                if cur_q4 >= a_thr:
                    aidx += 1
                    batch_collect, epoch_step, eval_if_cur = _build(a_grid[aidx])
                    (out / "anneal.json").write_text(json.dumps({"idx": aidx}))
                    print(f"  ANNEAL advance -> alpha={a_grid[aidx]} "
                          f"(cur-α q4={cur_q4:.3f} >= {a_thr})")
            if ev["main"]["gate_q4"] > best_gate:
                best_gate = ev["main"]["gate_q4"]
                save_ckpt(out / "ckpt_best.npz", theta, adam_state, key,
                          up + 1, config, best_gate=best_gate)
                print(f"  new best gate_q4={best_gate:.3f} -> ckpt_best.npz")
            if (stop_gate is not None and ev["main"]["gate_q4"] >= stop_gate
                    and (not stop_slope_pos or ev["main"]["slope"] > 0)):
                save_ckpt(out / "ckpt.npz", theta, adam_state, key, up + 1, config,
                          best_gate=best_gate)
                print(f"stop criterion reached at update {up}; exiting")
                break
        if (up + 1) % config["ckpt_every"] == 0 or up + 1 == config["updates"]:
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, up + 1, config,
                      best_gate=best_gate)

    return theta, unravel
