# changeling

Breeding fixed-weight ANNs whose learning algorithm lives in activations, so they
adapt in-context to environments they never saw — then dissecting them with SAEs to
extract the kernel. **Read `SPEC.md` first**; `PREREG_P0.md` locks the current phase.

## Layout

- `changeling/envs.py` — Phase 0 envs (bandit-8, catch), pure JAX, fixed interface
- `changeling/agent.py` — S1 GRU substrate
- `changeling/rollout.py` — RL² lifetime rollout + C4/C5 controls
- `changeling/es.py` — Route R1: OpenES (mirrored, rank-shaped, Adam), no deps
- `changeling/evaluate.py` — held-out gate metrics + control suite
- `changeling/train.py`, `run_p0.py` — loop, jsonl logs, npz checkpoint/resume

Dependencies: `jax` only (CPU jax for local smoke; Kaggle has GPU jax preinstalled).

## Run

```bash
# smoke (local CPU, minutes)
python run_p0.py --env bandit --out runs/smoke --gens 200 --hidden 32 --pop 128 --lifetimes 4
# gate config (Kaggle GPU; resume across sessions)
python run_p0.py --env bandit --out runs/gate_bandit
python run_p0.py --env bandit --out runs/gate_bandit --resume runs/gate_bandit/ckpt.npz
```

Gate 0 reads from the eval rows of `runs/*/log.jsonl`: `main.gate_q4` (bandit ≥ 0.85,
catch ≥ 0.90), `main.slope` > 0, controls `c4/c5 gate_q4` ≤ 0.225 (bandit).

## Kaggle (box-independent gate runs)

The gate runs as a **batch script kernel** built by `scripts/build_kernel.py`, which
concatenates the package into the single self-contained `kaggle/kernel.py` (no dataset
dependency, jax preinstalled). Once the push is verified running, the run is fully
server-side — the local box can disappear.

```bash
.venv/bin/python scripts/build_kernel.py     # rebuild after any package edit
kaggle kernels push -p kaggle                # then VERIFY within 90s (silent drops!)
kaggle kernels status asystemoffields/changeling-p0-gate
kaggle kernels output asystemoffields/changeling-p0-gate -p runs/kaggle_out
```

Survivability: the kernel exits cleanly (checkpointing) at a 7.5 h wall budget, stops
early when a gate criterion is met, and on relaunch resumes from a prior session's
output attached as a data source (`kernel_sources`/`dataset_sources` → matched by
`find_resume`). Local smoke of the built kernel: `CHANGELING_SMOKE=1 python kaggle/kernel.py`.
Budget tracking: PREREG G0-E, ≤ 24 GPU-h for all of Phase 0.
