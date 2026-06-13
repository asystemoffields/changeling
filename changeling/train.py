"""Generation loop with jsonl logging and npz checkpoint/resume."""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .agent import init_gru
from .looped import init_looped, make_step
from .interface import make_iface_fn
from .es import adam_init, flatten_params, make_gen_step
from .evaluate import full_eval
from .envs import ENVS
from .rollout import FITNESS


def init_core(key, config):
    """The R1 substrate params: looped core (halt+value heads) when config['loop'],
    else the plain GRU. The looped path is a strict superset (K_max=1 ≡ GRU)."""
    if config.get("loop", False):
        return init_looped(key, hidden=config["hidden"])
    return init_gru(key, hidden=config["hidden"])


def save_ckpt(path, theta, adam_state, key, gen, config, best_gate=-1.0):
    np.savez(
        path,
        theta=np.asarray(theta),
        adam_m=np.asarray(adam_state["m"]),
        adam_v=np.asarray(adam_state["v"]),
        adam_t=np.asarray(adam_state["t"]),
        key=np.asarray(key),
        gen=gen,
        config=json.dumps(config),
        best_gate=np.float32(best_gate),
    )


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    adam_state = dict(m=jnp.asarray(z["adam_m"]), v=jnp.asarray(z["adam_v"]),
                      t=jnp.int32(z["adam_t"]))
    # best_gate persisted since the resume-clobbers-ckpt_best fix; default for
    # checkpoints written before it (back-compat with existing npz).
    best_gate = float(z["best_gate"]) if "best_gate" in z.files else -1.0
    return (jnp.asarray(z["theta"]), adam_state, jnp.asarray(z["key"]),
            int(z["gen"]), json.loads(str(z["config"])), best_gate)


# Keys allowed to differ across a resume: survivability / logging cadence knobs
# that do NOT define the objective or training distribution. Everything else must
# match the saved config, or the resume silently optimizes a different problem
# from the saved theta (and the next ckpt overwrites the provenance).
_RESUME_FREE = {"max_seconds", "stop_gate", "stop_slope_pos", "updates", "gens",
                "out", "log_every", "eval_every", "ckpt_every"}


def assert_resume_cfg(saved_cfg, config):
    for k in config:
        if k in _RESUME_FREE:
            continue
        assert saved_cfg.get(k) == config[k], (
            f"resume config mismatch on {k!r}: saved={saved_cfg.get(k)!r} "
            f"new={config[k]!r} (objective-determining keys must match)")


def assert_pure_gate(config, eval_env):
    """If the training distribution is non-uniform (a curriculum: mix>0 or
    needle), the eval/gate env MUST be the pure gate task — otherwise gate_q4 is
    silently graded on an easier distribution (inflation, possible false PASS)."""
    ek = config.get("env_kwargs", {})
    if ek.get("mix", 0) > 0 or ek.get("needle", False):
        eek = config.get("eval_env_kwargs", None)
        assert (eek is not None and eek.get("mix", 0) == 0
                and not eek.get("needle", False)), (
            f"training distribution is non-uniform (env_kwargs={ek}) but "
            f"eval_env_kwargs={eek} is not the pure gate task")


def train(config, resume=None):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    env = ENVS[config["env"]](**config.get("env_kwargs", {}))
    # eval/gate listens to the pure gate task, which may differ from a curriculum
    # training distribution (mirrors ppo.py so ES grades correctly under mix>0).
    eval_env = ENVS[config["env"]](**config.get("eval_env_kwargs",
                                                config.get("env_kwargs", {})))
    assert_pure_gate(config, eval_env)

    step = make_step(config)
    ponder_cost = config.get("ponder_cost", 0.0)
    iface_fn = make_iface_fn(config, for_eval=False)        # B2 train interfaces
    eval_iface_fn = make_iface_fn(config, for_eval=True)    # held-out eval interfaces

    if resume:
        theta, adam_state, key, start_gen, saved_cfg, best_gate = load_ckpt(resume)
        assert_resume_cfg(saved_cfg, config)
        _, unravel = flatten_params(
            init_core(jax.random.PRNGKey(0), config))
        print(f"resumed gen={start_gen} from {resume} (best_gate={best_gate:.3f})")
    else:
        key = jax.random.PRNGKey(config["seed"])
        key, ki = jax.random.split(key)
        params = init_core(ki, config)
        theta, unravel = flatten_params(params)
        adam_state = adam_init(theta.shape[0])
        start_gen = 0
        best_gate = -1.0
        # C1 control: the random-init agent's eval, logged once up front
        c1 = full_eval(eval_env, unravel(theta), n=config["eval_n"],
                       seed=config["seed"], step=step, iface_fn=eval_iface_fn)
        _log(out, dict(gen=-1, condition="c1_random_init", **_flat(c1)))

    gen_step = make_gen_step(env, unravel, config["pop"],
                             config["n_lifetimes"], config["sigma"],
                             config["lr"],
                             fitness_fn=FITNESS[config.get("fitness", "late")],
                             step=step, ponder_cost=ponder_cost, iface_fn=iface_fn)
    print(f"env={env['name']} dim={theta.shape[0]} pop={config['pop']} "
          f"M={config['n_lifetimes']} T={env['T']}")
    print(f"R1 grades gate on eval_env={eval_env['name']} "
          f"kwargs={config.get('eval_env_kwargs', config.get('env_kwargs', {}))}")

    # survivability knobs (both optional, json-safe):
    #   max_seconds    — exit cleanly (checkpoint saved) before a session cap
    #   stop_gate      — stop once eval main.gate_q4 >= this
    #   stop_slope_pos — additionally require eval main.slope > 0 to stop
    max_seconds = config.get("max_seconds")
    stop_gate = config.get("stop_gate")
    stop_slope_pos = config.get("stop_slope_pos", False)

    t0 = time.time()
    for gen in range(start_gen, config["gens"]):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen, config,
                      best_gate=best_gate)
            print(f"wall-clock budget reached at gen {gen}; checkpointed, exiting")
            break
        key, kg = jax.random.split(key)
        theta, adam_state, stats = gen_step(theta, adam_state, kg)
        if (gen + 1) % config["log_every"] == 0:
            row = dict(gen=gen, sec=round(time.time() - t0, 1),
                       **{k: float(v) for k, v in stats.items()})
            _log(out, row)
            print(row)
        if (gen + 1) % config["eval_every"] == 0 or gen + 1 == config["gens"]:
            ev = full_eval(eval_env, unravel(theta), n=config["eval_n"],
                           seed=config["seed"], step=step, iface_fn=eval_iface_fn)
            _log(out, dict(gen=gen, **_flat(ev)))
            print(f"  eval g{gen}: main={ev['main']} "
                  f"c4={ev['c4_coin_reward']['gate_q4']:.3f} "
                  f"c5={ev['c5_no_memory']['gate_q4']:.3f}")
            if ev["main"]["gate_q4"] > best_gate:
                best_gate = ev["main"]["gate_q4"]
                save_ckpt(out / "ckpt_best.npz", theta, adam_state, key,
                          gen + 1, config, best_gate=best_gate)
                print(f"  new best gate_q4={best_gate:.3f} -> ckpt_best.npz")
            if (stop_gate is not None and ev["main"]["gate_q4"] >= stop_gate
                    and (not stop_slope_pos or ev["main"]["slope"] > 0)):
                save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen + 1, config,
                          best_gate=best_gate)
                print(f"stop criterion reached at gen {gen}; checkpointed, exiting")
                break
        if (gen + 1) % config["ckpt_every"] == 0 or gen + 1 == config["gens"]:
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen + 1, config,
                      best_gate=best_gate)

    return theta, unravel


def _flat(ev):
    return {f"{cond}.{k}": v for cond, d in ev.items() for k, v in d.items()}


def _log(out, row):
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
