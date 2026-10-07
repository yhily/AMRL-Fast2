# AT-4-4-4-v2 reproduction audit

This audit concerns only the AT-4-4-4-v2 operator label and its executable
implementation. It does not modify the AMRL learning objective or training
protocol.

## 1. What the public AMRL-Fast repository actually contains

The sole historical version of
`AMRL-FAST-META-NET/src /amrl_fast/alphatensor_ops.py` defines the string
`AT-4-4-4-v2` in `ALPHATENSOR_ALGORITHMS`, with metadata (`dimensions`,
claimed `speedup`, and `stability`) only.

Its executable chain is:

```text
OptimizedLSTMCell.forward
  -> alphatensor_matmul(..., algorithm='AT-4-4-4-v2')
  -> _simulated_alphatensor_matmul(...)
  -> torch.matmul(...)
```

Moreover, `check_alphatensor_availability()` returns `False` unconditionally,
so the public AMRL-Fast training path disables AlphaTensor before this call can
be used. There is no alternative commit, branch, tag, CUDA source, Triton
source, C++ extension, generated PTX, binary extension, or kernel-generation
script in that repository.

## 2. Public AlphaTensor code that is available

The official DeepMind AlphaTensor repository contains
`benchmarking/factorizations.py::get_4x4x4_alphatensor_gpu()`: a rank-49
factorization tailored to a **V100** benchmark of square 8192-by-8192
float32 matrices. Its official executable benchmark is a JAX-jitted reference
in `benchmarking/utils.py`; it is not a custom CUDA kernel. The documentation
states the reported benchmark is for a V100 and 8192-by-8192 matrices.

`amrl_fast/alphatensor_kernel.py` now executes these exact public V100 factors
under the separate name `AT-4-4-4-public-v100`. It performs block
decomposition and rank-49 recombination with vectorized PyTorch operations,
with no Python loop over ranks or tiles. It preserves PyTorch autograd.

`AT-4-4-4-v2` is intentionally fail-closed in that module. The name cannot be
mapped silently to the public V100 reference because the public AMRL-Fast code
does not establish that they are the same algorithm or deployment kernel.

## 3. Current execution and bottleneck evidence

The actual recurrent hidden-to-hidden shape in the public AMRL-Fast code is
`[batch, 256] @ [256, 1024]` for the combined gates (or the reconstructed
per-gate boundary `[batch, 256] @ [256, 256]`). At one controller decision,
the batch is one. Thus the relevant per-gate shape is `1x256 @ 256x256`, not a
square 256-by-256 multiplication.

The public rank-49 reference path contains:

1. zero-padding and 4-by-4 block layout;
2. left factor linear combination (`einsum`);
3. right factor linear combination (`einsum`);
4. 49 batched block products (`bmm`);
5. output recombination (`einsum`);
6. allocation, views, and autograd bookkeeping for these operations.

These are separate PyTorch GPU launches; there is no 4-by-4 tile fusion, no
fused epilogue, no CUDA graph capture by default, and no custom backward
kernel. For a one-row input, padding expands the row dimension from one to
four and the rank reduction designed for cubic multiplication does not reduce
the same work as a vendor GEMV. A vendor `torch.matmul` directly invokes an
optimized dense GEMV/GEMM path, whereas the reference path must execute the
additional transformations and 49 block products.

Local RTX 4050 Laptop GPU diagnostic (CUDA capability 8.9, five warmups and
25 synchronized measurements) produced the following *operator-only*
results. They are diagnostics, not manuscript data:

| Shape | Standard | Public V100 factor reference | Speedup |
|---|---:|---:|---:|
| `1x256 @ 256x256` | 0.811 ms | 1.627 ms | 0.498x |
| `256x256 @ 256x256` | 0.159 ms | 2.538 ms | 0.063x |

The earlier Titan XP measurement of the prior generic-factor PyTorch reference
was 0.228x. That path also passed output and gradient correlation checks but
was correctly rejected by the >=1.02x admission gate.

Run `scripts/at444_diagnose.py` to reproduce the stage-level timing on any
target GPU. It writes only JSON to stdout and never creates experiment results.

## 4. Missing v2 artifact

To reproduce a reportable `AT-4-4-4-v2` result, the following are absent from
the public AMRL-Fast repository:

- the actual v2 factor coefficients or generated arithmetic schedule;
- a CUDA, Triton, or C++ implementation that fuses the 4-by-4 factorization;
- the target architecture, tile size, launch parameters, and autotuning data;
- a build script or compatible binary extension;
- a backward implementation or an explicit autograd contract;
- evidence that the measured target shape is the controller's real batch and
  gate shape rather than a standalone square-matrix microbenchmark.

The current Titan XP has compute capability 6.1, while current Triton support
requires a newer architecture. The server also exposes CUDA runtime directories
but no `nvcc` command, so a CUDA extension cannot be built there until a CUDA
toolkit/compiler is installed or a matching prebuilt binary is supplied.

## 5. Minimal valid repair

1. Obtain the exact `AT-4-4-4-v2` source/binary and the factor schedule used
   for the manuscript's timing claim.
2. Add one backend module exposing
   `matmul(left, right, algorithm='AT-4-4-4-v2')`, with forward and backward
   supported by the supplied extension or a documented custom autograd
   function. Do not route this label through `torch.matmul`.
3. Build it for the target GPU and invoke the existing `OperatorBackend`
   screen on the real `[batch, 256] @ [256, 256]` gate shape. The existing
   screen already checks output/gradient fidelity, finite values, and a
   five-warmup/25-measurement synchronized >=1.02x condition.
4. Only if that screen admits the backend, run the paired five-seed Hopper and
   Walker experiments. Otherwise no AMRL-Fast runtime result is reportable.

This is the smallest repair that maintains the paper's operator-substitution
claim and prevents a reference implementation or fallback from becoming an
unjustified speedup result.
