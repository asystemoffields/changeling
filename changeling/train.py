"""Generation loop with jsonl logging and npz checkpoint/resume."""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .agent import init_gru
from .es import adam_init, flatten_params, make_gen_step
from .evaluate import full_eval
from .envs import ENVS


def save_ckpt(path, theta, adam_state, key, gen, config):
    np.savez(
        path,
        theta=np.asarray(theta),
        adam_m=np.asarray(adam_state["m"]),
        adam_v=np.asarray(adam_state["v"]),
        adam_t=np.asarray(adam_state["t"]),
        key=np.asarray(key),
        gen=gen,
        config=json.dumps(config),
    )


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    adam_state = dict(m=jnp.asarray(z["adam_m"]), v=jnp.asarray(z["adam_v"]),
                      t=jnp.int32(z["adam_t"]))
    return (jnp.asarray(z["theta"]), adam_state, jnp.asarray(z["key"]),
            int(z["gen"]), json.loads(str(z["config"])))


def train(config, resume=None):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    env = ENVS[config["env"]](**config.get("env_kwargs", {}))

    if resume:
        theta, adam_state, key, start_gen, saved_cfg = load_ckpt(resume)
        for k in ("env", "hidden", "pop", "sigma", "lr", "n_lifetimes"):
            assert saved_cfg[k] == config[k], f"resume config mismatch on {k}"
        _, unravel = flatten_params(
            init_gru(jax.random.PRNGKey(0), hidden=config["hidden"]))
        print(f"resumed gen={start_gen} from {resume}")
    else:
        key = jax.random.PRNGKey(config["seed"])
        key, ki = jax.random.split(key)
        params = init_gru(ki, hidden=config["hidden"])
        theta, unravel = flatten_params(params)
        adam_state = adam_init(theta.shape[0])
        start_gen = 0
        # C1 control: the random-init agent's eval, logged once up front
        c1 = full_eval(env, unravel(theta), n=config["eval_n"], seed=config["seed"])
        _log(out, dict(gen=-1, condition="c1_random_init", **_flat(c1)))

    from .rollout import FITNESS
    gen_step = make_gen_step(env, unravel, config["pop"],
                             config["n_lifetimes"], config["sigma"],
                             config["lr"],
                             fitness_fn=FITNESS[config.get("fitness", "late")])
    print(f"env={env['name']} dim={theta.shape[0]} pop={config['pop']} "
          f"M={config['n_lifetimes']} T={env['T']}")

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
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen, config)
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
            ev = full_eval(env, unravel(theta), n=config["eval_n"],
                           seed=config["seed"])
            _log(out, dict(gen=gen, **_flat(ev)))
            print(f"  eval g{gen}: main={ev['main']} "
                  f"c4={ev['c4_coin_reward']['gate_q4']:.3f} "
                  f"c5={ev['c5_no_memory']['gate_q4']:.3f}")
            if (stop_gate is not None and ev["main"]["gate_q4"] >= stop_gate
                    and (not stop_slope_pos or ev["main"]["slope"] > 0)):
                save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen + 1, config)
                print(f"stop criterion reached at gen {gen}; checkpointed, exiting")
                break
        if (gen + 1) % config["ckpt_every"] == 0 or gen + 1 == config["gens"]:
            save_ckpt(out / "ckpt.npz", theta, adam_state, key, gen + 1, config)

    return theta, unravel


def _flat(ev):
    return {f"{cond}.{k}": v for cond, d in ev.items() for k, v in d.items()}


def _log(out, row):
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
