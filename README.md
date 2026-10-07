# AMRL-Fast

Reference implementation for **AMRL-Fast**: a profiling-guided, stability-guarded
operator-substitution framework that reduces the outer-loop overhead of dual-loop
adaptive systems (instantiated on Adaptive Meta-Reinforcement Learning) without
changing the learning objective or the meta-controller logic.

AMRL-Fast combines hierarchical profiling, ROI-informed candidate prioritization,
output- and gradient-level fidelity screening, staged activation, and runtime
fallback. On a single-GPU budget it reduces meta-learning overhead from 35.1% to
8.1% and yields a 20.0% end-to-end wall-clock speedup, while adaptation-channel
diagnostics remain statistically indistinguishable from the baseline
($I(U;\Delta R)$ = 0.834 vs. 0.836 bits, $p = 0.68$, five-seed protocol) and
measured adaptation information throughput improves by 24%.

## Repository layout

```text
amrl-fast-linux/          # Main experiment package (Linux / CUDA)
├── amrl_fast/            # Core library: agent, envs, experiment, kernels,
│                         # LSTM meta-controller, statistics, telemetry
├── configs/              # One YAML config per environment / protocol
├── scripts/              # Benchmark, preflight, audit and probe entry points
├── tests/                # Unit tests
├── artifacts/            # Provenance and audit notes
├── requirements-*.txt    # Pinned environment variants
└── pyproject.toml
third_party/alphatensor/  # DeepMind AlphaTensor reference factorization
                          # (required reference benchmark route, see below)
```

## Installation

```bash
cd amrl-fast-linux
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-linux.txt
```

`Hopper-v2`, `Walker2d-v2` and `HalfCheetah-v2` are legacy Gym IDs. On a modern
Gymnasium/MuJoCo stack, pass `--allow-version-fallback` to resolve to `-v4`;
every result file records the resolved ID. Do not present a `-v4` result as
`-v2` without making that version change explicit.

For an exact legacy `-v2` attempt, additionally install:

```bash
pip install -r requirements-legacy-mujoco-v2.txt
```

## Reproducing the results

Run the paired one-seed preflight first. It writes raw delayed-MI pairs, reward
curves, component timings, fidelity and forced-fallback checks to
`artifacts/preflight/`, and marks every output as development-only:

```bash
python scripts/preflight_validation.py \
  --environments Hopper-v2 Walker2d-v2 \
  --seed 0 --timesteps 1024 --adaptation-interval 128 \
  --evaluation-episodes 2 --allow-version-fallback --device cuda
```

Do not start the five-seed runs unless `preflight_acceptance.json` reports
`all_checks_pass: true` for both environments.

```bash
python scripts/run_benchmark.py --config configs/hopper-v2.yaml \
  --allow-version-fallback --device cuda
python scripts/run_benchmark.py --config configs/walker2d-v2.yaml \
  --allow-version-fallback --device cuda
```

Each completed paired five-seed run writes
`results/<environment>/{runtime,performance,adaptation,profiling,stability}.json`:

- `runtime.json` — AMRL / AMRL-Fast wall-clock time and relative reduction
- `performance.json` — final returns over 20 evaluation episodes per seed
- `adaptation.json` — discrete joint plug-in $I(U;\Delta R)$, paired seed-level
  $p$-value, paired Cohen's $d_z$
- `profiling.json` — AMRL component times
- `stability.json` — fallback count/ratio and NaN/Inf count for AMRL-Fast

No JSON results are pre-populated: all numbers must come from actual runs.

## Execution backend

`AMRL-Fast-Rebuilt-Kernel` is the reportable execution candidate family. It
targets the controller's real combined four-gate LSTM tensors
(`[B,5] @ [1024,5]^T` and `[B,256] @ [1024,256]^T`) through PyTorch's native
ATen LSTM-cell forward/backward implementation, and also screens an `addmm`
four-gate route. The fastest route that passes the same fidelity and speed
criteria is selected per GPU.

The separately identified public DeepMind 4x4x4 rank-49 factorization is used
**only** as the required reference benchmark route:

```bash
git clone https://github.com/google-deepmind/alphatensor.git third_party/alphatensor
```

`scripts/benchmark_rebuilt_lstm.py` measures the manual controller operation,
the public rank-49 reference, and the rebuilt four-gate backend, recording
forward, backward and forward+backward latencies plus output/gradient fidelity.

The rebuilt backend is admitted before every Fast seed only when output
correlation $\ge$ 0.999, gradient correlation $\ge$ 0.99, relative L2 error
$\le$ 5e-3, max absolute error $\le$ 1e-1, all values are finite, and the
measured forward+backward speedup is at least 1.02x. A rejected backend stops
the reportable Fast run.

## Tests

```bash
cd amrl-fast-linux && python -m pytest
```

## Notes on scope

AMRL-Fast is a deployment-decision framework for execution-layer substitution in
dual-loop adaptive workloads, not a universally superior runtime optimizer. It is
most effective when computationally dominant outer-loop operators are repeatedly
invoked with stable tensor shapes; workloads dominated by dynamic dimensions,
irregular operators, or extreme numerical sensitivity are less directly
applicable.

See `amrl-fast-linux/README.md` and
`amrl-fast-linux/artifacts/AMRL_PUBLIC_CODE_ARCHAEOLOGY.md` for reconstruction
provenance and audit details.
