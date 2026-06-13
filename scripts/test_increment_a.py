"""Increment A integration gates (SPEC §5b/S3-M.1; reports/memory_arch_risks.md
"First build increment"). All un-metered / CPU. These MUST pass before any
metered run and before the PREREG_P1 lock.

  Gate 1  LYNCHPIN — replay exactness: collect()'s stored commit-logp == the
          loss forward's recomputed commit-logp (<1e-5). FAIL ⇒ the per-step fn
          is not deterministic-pure ⇒ the PPO ratio silently diverges.
  Gate 2  Regression safety: (a) loop=False rollout is BITWISE-identical to the
          literal pre-B1 gru rollout (so G0-A=0.6216 reproduces exactly); (b) the
          looped substrate at K_max=1 reduces to a single gru step.
  Gate 3  Gradient/CRN sanity: one ES gen_step + one PPO epoch_step (both routes)
          are NaN-free with finite gnorm; member-fitness is bitwise-deterministic
          given fixed roll_keys (antithetic CRN survives marginalization).
  Gate 4  Loop is exercised: mean E[K] > 1.5 on a looped smoke (K-collapse tripwire).

Run:  .venv/bin/python -m scripts.test_increment_a
"""
import jax
import jax.numpy as jnp

from changeling import N_ACT, IN_DIM
from changeling.agent import init_gru, gru_step, hidden_size
from changeling.looped import init_looped, make_step, micro_loop
from changeling.rollout import rollout, late_weighted_fitness
from changeling.evaluate import full_eval
from changeling.envs import ENVS
from changeling.es import make_gen_step, adam_init, flatten_params
from changeling.ppo import init_ppo_params, collect, make_update_step

OK = []
def chk(name, cond, extra=""):
    cond = bool(cond)
    OK.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))


# ----- a literal copy of the PRE-B1 ES rollout step (the regression reference) --
def old_rollout(env, params, task, key):
    """Verbatim pre-B1 rollout (gru_step + categorical(logits)); no c-controls,
    no step_fn abstraction. The byte-exact baseline G0-A was measured against."""
    T = env["T"]
    key, k0 = jax.random.split(key)
    state0, obs0 = env["reset"](k0, task)
    h0 = jnp.zeros(hidden_size(params))
    carry0 = (h0, state0, obs0, jnp.zeros(N_ACT), jnp.float32(0.0),
              jnp.float32(1.0), key)

    def step_fn(carry, _):
        h, state, obs, last_a, last_r, boundary, key = carry
        key, ka, kr, kres, kc4 = jax.random.split(key, 5)
        r_in = last_r
        x = jnp.concatenate([obs, last_a, jnp.array([r_in, boundary])])
        h, logits = gru_step(params, h, x)
        a = jax.random.categorical(ka, logits)
        state, obs, r, done, metric = env["step"](state, a, kr, task)
        rs_state, rs_obs = env["reset"](kres, task)
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done, new, old), rs_state, state)
        obs = jnp.where(done, rs_obs, obs)
        carry = (h, state, obs, jax.nn.one_hot(a, N_ACT), r,
                 done.astype(jnp.float32), key)
        return carry, (r, metric, done.astype(jnp.float32))

    _, (rewards, metrics, dones) = jax.lax.scan(step_fn, carry0, None, length=T)
    return rewards, metrics, dones


def main():
    key = jax.random.PRNGKey(0)
    H = 16
    env = ENVS["bandit"](n_arms=4, lifetime=24)
    lcfg = {"loop": True, "k_max": 4, "k_min": 2, "hidden": H}
    step_loop = make_step(lcfg)
    step_leg = make_step({})

    # ===== Gate 1 — LYNCHPIN: collect == forward, bitwise-pure ====================
    print("Gate 1 — replay exactness (collect commit-logp == forward commit-logp)")
    kp, kt, kc = jax.random.split(key, 3)
    pp = init_ppo_params(kp, H, lcfg)
    task = env["sample_task"](kt)
    xs, acts, logps, vals, rews = collect(env, pp, task, kc, 0.1, step_loop, True)

    def forward_logp(params_core, xs_one):
        h0 = jnp.zeros(H)
        def st(h, x):
            h, _ss, logp_all, _v, _aux = step_loop(params_core, h, x)
            return h, logp_all
        _, lpa = jax.lax.scan(st, h0, xs_one)
        return lpa

    lpa = forward_logp(pp["gru"], xs)
    logp_fwd = jnp.take_along_axis(lpa, acts[:, None], -1)[:, 0]
    d1 = float(jnp.max(jnp.abs(logp_fwd - logps)))
    chk("collect stored logp == forward recomputed logp (<1e-5)", d1 < 1e-5,
        f"max|Δ|={d1:.2e}")
    # value replay too (commit_v must be reconstructable)
    def forward_v(params_core, xs_one):
        h0 = jnp.zeros(H)
        def st(h, x):
            h, _ss, _lpa, v, _aux = step_loop(params_core, h, x)
            return h, v
        _, vv = jax.lax.scan(st, h0, xs_one)
        return vv
    dv = float(jnp.max(jnp.abs(forward_v(pp["gru"], xs) - vals)))
    chk("collect stored commit_v == forward recomputed commit_v (<1e-5)", dv < 1e-5,
        f"max|Δ|={dv:.2e}")

    # ===== Gate 2a — loop=False rollout BITWISE == pre-B1 gru rollout ============
    print("Gate 2a — legacy rollout regression (bitwise vs pre-B1 reference)")
    kpg, ktg, krg = jax.random.split(jax.random.PRNGKey(7), 3)
    pg = init_gru(kpg, hidden=H)
    tg = env["sample_task"](ktg)
    r_old, m_old, d_old = old_rollout(env, pg, tg, krg)
    r_new, m_new, d_new, e_new = rollout(env, pg, tg, krg, step=step_leg)
    chk("rewards bitwise-identical", float(jnp.max(jnp.abs(r_old - r_new))) == 0.0)
    chk("metrics bitwise-identical", float(jnp.max(jnp.abs(m_old - m_new))) == 0.0)
    chk("dones bitwise-identical", float(jnp.max(jnp.abs(d_old - d_new))) == 0.0)
    chk("legacy e_k ≡ 1.0", float(jnp.max(jnp.abs(e_new - 1.0))) == 0.0)
    # also: step=None default routes to the same legacy path
    r_def, *_ = rollout(env, pg, tg, krg)
    chk("rollout(step=None) == rollout(step=legacy)",
        float(jnp.max(jnp.abs(r_def - r_new))) == 0.0)

    # ===== Gate 2b — K_max=1 looped ≡ single gru step ===========================
    print("Gate 2b — K_max=1 reduction (looped ≡ gru)")
    kr1 = jax.random.PRNGKey(3)
    pl = init_looped(kr1, hidden=H)
    xprobe = jax.random.normal(jax.random.PRNGKey(4), (IN_DIM,))
    hprobe = jax.random.normal(jax.random.PRNGKey(5), (H,))
    h_g, logits_g = gru_step(pl, hprobe, xprobe)
    hn, clp, cv, aux = micro_loop(pl, hprobe, xprobe, k_max=1, k_min=2)
    chk("K_max=1 commit_logp == log_softmax(gru logits)",
        float(jnp.max(jnp.abs(clp - jax.nn.log_softmax(logits_g)))) < 1e-5)
    chk("K_max=1 h_new == gru hidden", float(jnp.max(jnp.abs(hn - h_g))) < 1e-6)

    # ===== Gate 3 — gradient / CRN sanity (both routes) =========================
    print("Gate 3 — gradient + CRN sanity")
    # ---- ES, looped ----
    es_cfg = dict(pop=8, n_lifetimes=4, sigma=0.05, lr=0.01)
    pl_es = init_looped(jax.random.PRNGKey(11), hidden=H)
    theta_l, unravel_l = flatten_params(pl_es)
    gen_step_l = make_gen_step(env, unravel_l, es_cfg["pop"], es_cfg["n_lifetimes"],
                               es_cfg["sigma"], es_cfg["lr"], step=step_loop,
                               ponder_cost=0.01)
    adam_l = adam_init(theta_l.shape[0])
    th2, ad2, stats = gen_step_l(theta_l, adam_l, jax.random.PRNGKey(12))
    finite_es = all(bool(jnp.isfinite(v).all()) for v in stats.values()) and \
        bool(jnp.isfinite(th2).all())
    chk("ES looped gen_step finite (stats + theta)", finite_es,
        f"fit_mean={float(stats['fit_mean']):.4f}")
    # CRN: same theta, same roll_keys -> bitwise-identical rollout reward/fitness
    kt_crn, kr_crn = jax.random.split(jax.random.PRNGKey(99))
    tasks_crn = jax.vmap(env["sample_task"])(jax.random.split(kt_crn, 4))
    rkeys_crn = jax.random.split(kr_crn, 4)
    def member_fit(theta_flat):
        params = unravel_l(theta_flat)
        def one(t, k):
            rw, _, _, ek = rollout(env, params, t, k, step=step_loop)
            return late_weighted_fitness(rw) - 0.01 * ek.mean()
        return jax.vmap(one)(tasks_crn, rkeys_crn).mean()
    f_a, f_b = float(member_fit(theta_l)), float(member_fit(theta_l))
    chk("CRN: member_fitness bitwise-deterministic given fixed roll_keys",
        f_a == f_b, f"{f_a!r}")
    # ---- ES, legacy (ponder_cost=0) ----
    pg_es = init_gru(jax.random.PRNGKey(13), hidden=H)
    theta_g, unravel_g = flatten_params(pg_es)
    gen_step_g = make_gen_step(env, unravel_g, es_cfg["pop"], es_cfg["n_lifetimes"],
                               es_cfg["sigma"], es_cfg["lr"], step=step_leg,
                               ponder_cost=0.0)
    thg2, _, statsg = gen_step_g(theta_g, adam_init(theta_g.shape[0]),
                                 jax.random.PRNGKey(14))
    chk("ES legacy gen_step finite", bool(jnp.isfinite(thg2).all()))

    # ---- PPO, looped ----
    ppo_cfg = dict(loop=True, k_max=4, k_min=2, hidden=H, n_lifetimes=4,
                   gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01,
                   lr=3e-4, max_grad_norm=1.0, lam_p=0.2, kl_coef=0.01,
                   reward_scale=0.05)
    pp_l = init_ppo_params(jax.random.PRNGKey(21), H, ppo_cfg)
    theta_pl, unravel_pl = flatten_params(pp_l)
    bc, es_step = make_update_step(env, unravel_pl, ppo_cfg)
    batch = bc(theta_pl, jax.random.PRNGKey(22))
    th_pl2, ad_pl2, loss_pl, aux_pl = es_step(theta_pl, adam_init(theta_pl.shape[0]),
                                              batch)
    finite_ppo = bool(jnp.isfinite(loss_pl)) and bool(jnp.isfinite(th_pl2).all()) \
        and bool(jnp.isfinite(aux_pl["gnorm"]))
    chk("PPO looped epoch_step finite (loss/theta/gnorm)", finite_ppo,
        f"loss={float(loss_pl):.4f} kl={float(aux_pl['kl']):.4f} "
        f"E[K]={float(aux_pl['e_k']):.3f}")
    chk("PPO looped reports kl + e_k in aux", "kl" in aux_pl and "e_k" in aux_pl)
    # ---- PPO, legacy ----
    leg_cfg = dict(loop=False, hidden=H, n_lifetimes=4, gamma=0.99, lam=0.95,
                   clip=0.2, vf_coef=0.5, ent_coef=0.01, lr=3e-4,
                   max_grad_norm=1.0, reward_scale=0.05)
    pp_g = init_ppo_params(jax.random.PRNGKey(31), H, leg_cfg)
    theta_pg, unravel_pg = flatten_params(pp_g)
    bcg, es_stepg = make_update_step(env, unravel_pg, leg_cfg)
    batchg = bcg(theta_pg, jax.random.PRNGKey(32))
    th_pg2, _, loss_pg, aux_pg = es_stepg(theta_pg, adam_init(theta_pg.shape[0]),
                                          batchg)
    chk("PPO legacy epoch_step finite", bool(jnp.isfinite(loss_pg)) and
        bool(jnp.isfinite(th_pg2).all()))
    chk("PPO legacy aux has NO kl key (exact legacy graph)", "kl" not in aux_pg)

    # ===== Gate 4 — loop is exercised (E[K] > 1.5) ==============================
    print("Gate 4 — adaptive-K exercised (E[K] tripwire)")
    ev = full_eval(env, pl_es, n=32, seed=0, step=step_loop)
    ek_mean = ev["main"]["e_k_mean"]
    ek_q4 = ev["main"]["e_k_q4"]
    chk("mean E[K] > 1.5 on looped smoke", ek_mean > 1.5,
        f"E[K]_mean={ek_mean:.3f} E[K]_q4={ek_q4:.3f} (k_max=4)")
    # legacy eval reports E[K]==1
    ev_leg = full_eval(env, pg, n=16, seed=0, step=step_leg)
    chk("legacy eval E[K] == 1.0", abs(ev_leg["main"]["e_k_mean"] - 1.0) < 1e-9)

    print(f"\n{sum(OK)}/{len(OK)} gates passed")
    import sys
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
