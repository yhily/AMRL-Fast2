# AMRL-Fast Linux continuous-control extension

This is a normalized, Linux-oriented experiment package prepared from the
public AMRL-Fast repository.  It keeps the reported PPO base learner, 256-unit
LSTM meta-controller, 10k adaptation cadence, warmup/transition schedule,
bounded hyperparameter updates, fidelity gate, and one-cycle fallback policy.
It adds only a shared environment factory and paper-format experiment logging
for `Hopper-v2` and `Walker2d-v2`.

## Rebuilt execution backend

`AMRL-Fast-Rebuilt-Kernel` is the reportable execution candidate family in
this package. It targets the controller's real combined
four-gate LSTM tensors (`[B,5] @ [1024,5]^T` and
`[B,256] @ [1024,256]^T`) through PyTorch's native ATen LSTM-cell
forward/backward implementation, and also screens an `addmm` four-gate route.
The fastest route that passes the same fidelity and speed criteria is selected
per GPU. The controller equations, parameters, PPO objective, training budget,
fidelity criteria, and fallback policy are not changed.

The separately identified public DeepMind 4x4x4 rank-49 factorization remains
available only as the required reference benchmark route. Obtain it once:

```bash
cd ..
git clone https://github.com/google-deepmind/alphatensor.git third_party/alphatensor
cd amrl-fast-linux
```

`scripts/benchmark_rebuilt_lstm.py` measures three routes on the real shape:
the public manual controller operation, the public rank-49 reference, and the
rebuilt four-gate backend. It records separate forward, backward, and
forward-plus-backward latencies with CUDA warmup/synchronization, plus output
and gradient fidelity. The rebuilt backend is admitted before every Fast seed
only when output correlation >= 0.999, gradient correlation >= 0.99, relative
L2 error <= 5e-3, max absolute error <= 1e-1, all values are finite, and its
measured forward-plus-backward speedup is at least 1.02x. A rejected backend
stops the reportable Fast run.

To isolate the operator cost before using a GPU server, run:

```bash
python scripts/benchmark_rebuilt_lstm.py --device cuda \
  --output artifacts/at_rebuild/benchmark_linux.json
python scripts/smoke_rebuilt_lstm.py --environment Hopper-v2 \
  --allow-version-fallback --device cuda \
  --output artifacts/at_rebuild/hopper_integration.json
python scripts/profile_rebuilt_lstm.py --device cuda \
  --output artifacts/at_rebuild/profile_linux.json
```

These are development-only records and cannot be copied into manuscript
tables. The normal five-seed runner performs a fresh per-seed admission screen
and records the corresponding report in every JSON result.

## Linux installation

```bash
cd amrl-fast-linux
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-linux.txt
```

`Hopper-v2`, `Walker2d-v2`, and `HalfCheetah-v2` are legacy Gym IDs.  On a
modern Gymnasium/MuJoCo installation, use the explicit `--allow-version-fallback`
flag to resolve to `-v4`; every result file records the resolved ID.  Do not
present a `-v4` result as `-v2` without making that version change explicit.

For an exact legacy `-v2` attempt, install the server-compatible legacy stack
in addition to the main requirements:

```bash
pip install -r requirements-legacy-mujoco-v2.txt
```

## Run the new benchmarks

Before a formal five-seed run, execute the paired one-seed preflight. It
reuses the formal seed runner, writes raw delayed MI pairs, reward curves,
component timings, fidelity, and forced fallback-path checks under
`artifacts/preflight/`, and marks every output as development-only:

```bash
python scripts/preflight_validation.py \
  --environments Hopper-v2 Walker2d-v2 \
  --seed 0 --timesteps 1024 --adaptation-interval 128 \
  --evaluation-episodes 2 --allow-version-fallback --device cuda
```

Do not begin the five-seed runs unless `preflight_acceptance.json` reports
`all_checks_pass: true` for both environments.

```bash
python scripts/run_benchmark.py --config configs/hopper-v2.yaml --allow-version-fallback --device cuda
python scripts/run_benchmark.py --config configs/walker2d-v2.yaml --allow-version-fallback --device cuda
```

These training commands do not render frames. On headless machines, leave
`MUJOCO_GL` unset unless the local MuJoCo/EGL stack has been independently
validated.

## Reconstruction provenance

The AMRL baseline repository provides PPO/LSTM components but does not connect
a valid outer optimiser to `MetaLearner`; its public training entry never calls
a meta optimiser. This package restores the stated architecture without adding
a new objective: the LSTM controller samples a bounded hyperparameter action,
the following adaptation window supplies its delayed meta reward, and the
stored transition is optimised by the repository's declared clipped PPO-style
outer update. AMRL and AMRL-Fast share that exact logic; the only conditional
branch is the admitted fused four-gate execution backend.

The public schedule provides only one 10k-step warmup action, so it cannot
produce a statistically meaningful distribution of PPO meta-gradient norms.
The runtime guard therefore remains in standard execution until two *observed*
finite outer-PPO gradients exist; it never manufactures a zero-gradient
three-sigma threshold. This conservative deferral is recorded through the
existing fallback counters and should be stated if the implementation details
are revised in the manuscript.

## Output

Each completed paired five-seed run writes:

```text
results/<requested environment>/
├── runtime.json
├── performance.json
├── adaptation.json
├── profiling.json
└── stability.json
```

`runtime.json` supplies AMRL / AMRL-Fast total wall-clock time and
`(T_AMRL - T_AMRL-Fast) / T_AMRL`.  `performance.json` contains final returns
from 20 evaluation episodes per seed.  `adaptation.json` reports the discrete
joint plug-in `I(U; ΔR)` estimate, paired seed-level p-value, and paired
Cohen's dz.  Evaluation episodes are never treated as independent replicates.
`profiling.json` contains the AMRL component times.  `stability.json` contains
the existing fallback count/ratio and NaN/Inf count for AMRL-Fast.

No JSON results are pre-populated: numerical values must come from the Linux
server's actual runs.
