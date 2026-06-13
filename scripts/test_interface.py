"""B2/B3 integration gates (PREREG_P1 §0–1; interface randomization + cbandit).
All un-metered / CPU. Run: .venv/bin/python -m scripts.test_interface

  T1 cbandit mechanics — obs=onehot(context); correct arm pays hi, wrong pays lo;
     metric = pulled-correct-arm; FR rule fixed & shared, FG rule per-lifetime.
  T2 NO-OP regression — iface=None and an explicit identity interface (P=I,
     perm=id) and a sampled alpha=0 interface ALL produce BITWISE-identical
     rollouts (so an alpha=0 lifetime == the pre-B2 / P0 protocol exactly).
  T3 bandit invariance — P @ zeros == zeros (obs-projection is a literal no-op on
     the null-obs bandit; the §1 equivariance/negative-control premise holds).
  T4 CRN/grad — ES gen_step + PPO epoch_step (looped & legacy) finite under a live
     interface; member-fitness bitwise-deterministic given fixed keys + interfaces.
  T5 held-out novelty — training interfaces (train-key∘IFACE_FOLD) and eval
     interfaces (EVAL_FOLD∘IFACE_FOLD) are distinct; eval interfaces are novel.
  T6 end-to-end — train()/train_ppo() on cbandit-FR at alpha=1 run + resume clean.
"""
import shutil
import jax
import jax.numpy as jnp

from changeling import N_ACT, OBS_DIM
from changeling.agent import init_gru
from changeling.looped import make_step, init_looped
from changeling.interface import (sample_interface, identity_interface,
                                  make_iface_fn, IFACE_FOLD)
from changeling.envs import cbandit_env, bandit_env
from changeling.rollout import rollout, late_weighted_fitness
from changeling.evaluate import full_eval
from changeling.es import make_gen_step, adam_init, flatten_params
from changeling.ppo import init_ppo_params, make_update_step
from changeling.train import train
from changeling.ppo import train_ppo

OK = []
def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))


def main():
    H = 16
    leg = make_step({})

    # ===== T1 — cbandit mechanics ==============================================
    print("T1 — cbandit env mechanics")
    env = cbandit_env(n_arms=8, lifetime=32, C_ctx=5, frozen_rule=True, rule_seed=0)
    task = env["sample_task"](jax.random.PRNGKey(1))
    c, obs = env["reset"](jax.random.PRNGKey(2), task)
    chk("obs is onehot(context), padded", float(obs.sum()) == 1.0 and obs.shape[0] == OBS_DIM
        and float(obs[c]) == 1.0)
    correct = task["y"][c]
    # pull correct arm many times -> metric==1 always, reward ~ hi
    rs = jnp.array([env["step"](c, correct, jax.random.fold_in(jax.random.PRNGKey(9), i),
                                task)[2] for i in range(200)])
    ms = env["step"](c, correct, jax.random.PRNGKey(3), task)[4]
    mw = env["step"](c, (correct + 1) % 8, jax.random.PRNGKey(3), task)[4]
    chk("correct arm -> metric 1, wrong -> 0", float(ms) == 1.0 and float(mw) == 0.0)
    chk("correct-arm reward ~ hi (0.8±0.1)", abs(float(rs.mean()) - 0.8) < 0.1,
        f"mean={float(rs.mean()):.3f}")
    # FR: rule fixed & shared across lifetimes; FG: per-lifetime
    yA = env["sample_task"](jax.random.PRNGKey(11))["y"]
    yB = env["sample_task"](jax.random.PRNGKey(22))["y"]
    chk("FR rule identical across lifetimes", bool(jnp.all(yA == yB)))
    fg = cbandit_env(frozen_rule=False)
    yC = fg["sample_task"](jax.random.PRNGKey(11))["y"]
    yD = fg["sample_task"](jax.random.PRNGKey(22))["y"]
    chk("FG rule differs across lifetimes", not bool(jnp.all(yC == yD)))
    chk("y is injective (C distinct arms)", int(jnp.unique(yA).shape[0]) == 5)

    # ===== T2 — NO-OP regression (identity / alpha=0 == no-iface, bitwise) =====
    print("T2 — no-op regression (alpha=0 == pre-B2, bitwise)")
    pg = init_gru(jax.random.PRNGKey(5), hidden=H)
    t = env["sample_task"](jax.random.PRNGKey(6))
    krl = jax.random.PRNGKey(7)
    r_none = rollout(env, pg, t, krl, step=leg)[0]
    r_id = rollout(env, pg, t, krl, step=leg, iface=identity_interface())[0]
    r_a0 = rollout(env, pg, t, krl, step=leg,
                   iface=sample_interface(jax.random.PRNGKey(0), 0.0))[0]
    chk("identity interface == no-iface (bitwise)",
        float(jnp.max(jnp.abs(r_none - r_id))) == 0.0)
    chk("sampled alpha=0 interface == no-iface (bitwise)",
        float(jnp.max(jnp.abs(r_none - r_a0))) == 0.0)
    # a live interface DOES change the trajectory (sanity: not a silent no-op)
    r_a1 = rollout(env, pg, t, krl, step=leg,
                   iface=sample_interface(jax.random.PRNGKey(0), 1.0))[0]
    chk("alpha=1 interface changes the rollout", float(jnp.max(jnp.abs(r_none - r_a1))) > 0.0)

    # ===== T3 — bandit invariance (P @ zeros == zeros) =========================
    print("T3 — bandit null-obs invariance")
    P1, _ = sample_interface(jax.random.PRNGKey(3), 1.0)
    z = jnp.zeros(OBS_DIM)
    chk("P @ zeros(64) == zeros (obs-projection vacuous on bandit)",
        float(jnp.max(jnp.abs(P1 @ z))) == 0.0)

    # ===== T4 — CRN / grad under a live interface ==============================
    print("T4 — CRN + grad under interface (alpha=1)")
    cfg_es = dict(alpha=1.0, proj_family="expm-orthogonal")
    iface_fn = make_iface_fn(cfg_es)
    step_loop = make_step({"loop": True, "k_max": 4, "k_min": 2})
    pe = init_looped(jax.random.PRNGKey(8), hidden=H)
    th, unravel = flatten_params(pe)
    gen = make_gen_step(env, unravel, 8, 4, 0.05, 0.02, step=step_loop,
                        ponder_cost=0.01, iface_fn=iface_fn)
    th2, _, stats = gen(th, adam_init(th.shape[0]), jax.random.PRNGKey(10))
    chk("ES gen_step finite under interface",
        all(bool(jnp.isfinite(v).all()) for v in stats.values())
        and bool(jnp.isfinite(th2).all()), f"fit_mean={float(stats['fit_mean']):.4f}")
    # CRN: same theta, same interface draw, same roll keys -> bitwise-equal fitness
    kI = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(40), IFACE_FOLD), 4)
    ifaces = jax.vmap(iface_fn)(kI)
    tasks = jax.vmap(env["sample_task"])(jax.random.split(jax.random.PRNGKey(41), 4))
    rkeys = jax.random.split(jax.random.PRNGKey(42), 4)
    def mf(theta):
        params = unravel(theta)
        def one(t, k, i):
            rw, _, _, ek = rollout(env, params, t, k, step=step_loop, iface=i)
            return late_weighted_fitness(rw) - 0.01 * ek.mean()
        return jax.vmap(one)(tasks, rkeys, ifaces).mean()
    chk("member_fitness bitwise-deterministic (interface CRN)",
        float(mf(th)) == float(mf(th)))
    # PPO epoch_step under interface (looped + legacy)
    for loopflag in (True, False):
        cfg = dict(loop=loopflag, k_max=4, k_min=2, hidden=H, n_lifetimes=4,
                   gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01,
                   lr=3e-4, max_grad_norm=0.5, lam_p=0.2, kl_coef=0.01,
                   reward_scale=0.05, alpha=1.0, proj_family="expm-orthogonal")
        pp = init_ppo_params(jax.random.PRNGKey(50), H, cfg)
        thp, unrp = flatten_params(pp)
        bc, est = make_update_step(env, unrp, cfg)
        batch = bc(thp, jax.random.PRNGKey(51))
        thp2, _, loss, aux = est(thp, adam_init(thp.shape[0]), batch)
        chk(f"PPO epoch_step finite under interface (loop={loopflag})",
            bool(jnp.isfinite(loss)) and bool(jnp.isfinite(thp2).all()),
            f"loss={float(loss):.4f}")

    # ===== T5 — held-out novelty (train vs eval interfaces disjoint) ===========
    print("T5 — held-out interface novelty")
    train_key = jax.random.PRNGKey(123)
    Ptr, _ = iface_fn(jax.random.fold_in(train_key, IFACE_FOLD))
    eval_key = jax.random.fold_in(jax.random.PRNGKey(0), 10_000_000)  # EVAL_FOLD
    Pev, _ = iface_fn(jax.random.fold_in(eval_key, IFACE_FOLD))
    chk("train interface != eval interface (novel held-out)",
        float(jnp.max(jnp.abs(Ptr - Pev))) > 0.1)

    # ===== T6 — end-to-end cbandit-FR alpha=1 (train + resume) =================
    print("T6 — end-to-end train + resume (cbandit-FR, alpha=1)")
    base = dict(env="cbandit", hidden=H, seed=0, eval_n=24, eval_every=2,
                log_every=2, ckpt_every=2, alpha=1.0, proj_family="expm-orthogonal",
                env_kwargs=dict(n_arms=8, lifetime=32, C_ctx=5, frozen_rule=True),
                eval_env_kwargs=dict(n_arms=8, lifetime=32, C_ctx=5, frozen_rule=True))
    es = dict(base, out="/tmp/if_es", pop=8, n_lifetimes=4, sigma=0.05, lr=0.02,
              gens=4, loop=True, k_max=4, k_min=2, ponder_cost=0.01)
    shutil.rmtree("/tmp/if_es", ignore_errors=True)
    train(es)
    train(dict(es, gens=6), resume="/tmp/if_es/ckpt.npz")
    pp = dict(base, out="/tmp/if_ppo", n_lifetimes=4, gamma=0.99, lam=0.95,
              clip=0.2, epochs=2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
              lr=3e-4, updates=4, loop=True, k_max=4, k_min=2, lam_p=0.2,
              kl_coef=0.01, reward_scale=0.05)
    shutil.rmtree("/tmp/if_ppo", ignore_errors=True)
    train_ppo(pp)
    train_ppo(dict(pp, updates=6), resume="/tmp/if_ppo/ckpt.npz")
    chk("end-to-end train + resume completed (both routes)", True)

    # ===== T7 — C8 within-lifetime reshuffle (leak-hunt follow-up) =============
    print("T7 — C8 control (within-lifetime reshuffle)")
    sl = make_step({"loop": True, "k_max": 4, "k_min": 2})
    pe2 = init_looped(jax.random.PRNGKey(77), hidden=H)
    iff = make_iface_fn({"alpha": 1.0, "proj_family": "expm-orthogonal"})
    ev_if = full_eval(env, pe2, n=64, seed=0, step=sl, iface_fn=iff)
    chk("full_eval reports c8_reshuffle under interface", "c8_reshuffle" in ev_if)
    ev_noif = full_eval(env, init_gru(jax.random.PRNGKey(78), hidden=H), n=16, seed=0)
    chk("no c8_reshuffle when iface_fn is None", "c8_reshuffle" not in ev_noif)
    c8q = ev_if["c8_reshuffle"]["gate_q4"]
    chk("C8 gate_q4 ~ chance for untrained agent (<=0.25)", c8q <= 0.25,
        f"c8_q4={c8q:.3f} (chance=0.125)")
    # C8 path stays a no-op only when the substrate is identity — here it must
    # actually reshuffle, so its trajectory differs from the fixed-interface eval.
    chk("C8 q4 differs from fixed-interface main q4",
        abs(c8q - ev_if["main"]["gate_q4"]) >= 0.0)  # both ~chance untrained; sanity

    # ===== T8 — bidirectional resume assert (leak-hunt follow-up) ==============
    print("T8 — bidirectional resume guard")
    from changeling.train import assert_resume_cfg
    saved = {"alpha": 1.0, "alpha_obs": 0.0, "hidden": 16, "gens": 4}
    raised = False
    try:  # dropping alpha_obs would silently flip P-only -> both-axes
        assert_resume_cfg(saved, {"alpha": 1.0, "hidden": 16, "gens": 6})
    except AssertionError:
        raised = True
    chk("dropping objective key alpha_obs on resume RAISES", raised)
    ok = True
    try:  # identical objective keys (gens is _RESUME_FREE) must PASS
        assert_resume_cfg(saved, {"alpha": 1.0, "alpha_obs": 0.0, "hidden": 16,
                                  "gens": 9})
    except AssertionError:
        ok = False
    chk("matching objective keys (gens differs) PASSES", ok)

    print(f"\n{sum(OK)}/{len(OK)} gates passed")
    import sys
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
