"""Paper-protocol information and seed-level statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import spearmanr, ttest_rel


def _equal_frequency_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return robust equal-frequency discrete labels without adding a new metric."""
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0]):
        return np.zeros(values.shape[0], dtype=int)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(values.shape[0], dtype=int)
    return np.digitize(values, edges[1:-1], right=True)


def discrete_mutual_information(controller_outputs: list[list[float]], reward_deltas: list[float]) -> float:
    """Joint plug-in I(U; ΔR) in bits, using the manuscript binning rule."""
    outputs = np.asarray(controller_outputs, dtype=float)
    targets = np.asarray(reward_deltas, dtype=float)
    if outputs.ndim != 2 or outputs.shape[0] != targets.shape[0] or len(targets) < 2:
        return 0.0
    n_bins = int(np.clip(np.ceil(len(targets) ** (1 / 3)), 2, 10))
    columns = [_equal_frequency_codes(outputs[:, index], n_bins) for index in range(outputs.shape[1])]
    joint_u = np.ravel_multi_index(columns, dims=(n_bins,) * outputs.shape[1])
    binned_r = _equal_frequency_codes(targets, n_bins)
    pairs = np.stack([joint_u, binned_r], axis=1)
    _, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    unique_u, counts_u = np.unique(joint_u, return_counts=True)
    unique_r, counts_r = np.unique(binned_r, return_counts=True)
    p_u = dict(zip(unique_u.tolist(), (counts_u / len(targets)).tolist()))
    p_r = dict(zip(unique_r.tolist(), (counts_r / len(targets)).tolist()))
    mi = 0.0
    for (u_code, r_code), count in zip(np.unique(pairs, axis=0), pair_counts):
        p_ur = count / len(targets)
        mi += p_ur * np.log2(p_ur / (p_u[int(u_code)] * p_r[int(r_code)]))
    return float(mi)


def information_diagnostics(
    controller_outputs: list[list[float]], reward_deltas: list[float], *, wall_clock_seconds: float, k: int = 5
) -> dict:
    """Persist the manuscript's discrete, kNN and Spearman MI diagnostics.

    This is reporting-only: it consumes the already paired one-cycle delayed
    controller outputs and reward deltas and never feeds back into training.
    """
    outputs = np.asarray(controller_outputs, dtype=float)
    targets = np.asarray(reward_deltas, dtype=float)
    n = int(targets.size)
    n_bins = int(np.clip(np.ceil(n ** (1 / 3)), 2, 10)) if n else 0
    discrete = discrete_mutual_information(controller_outputs, reward_deltas)
    knn_mi = 0.0
    spearman = 0.0
    effective_k = min(int(k), max(n - 1, 0))
    if n >= 3 and outputs.ndim == 2 and outputs.shape[0] == n:
        # Kraskov--Stögbauer--Grassberger estimator using the max norm.  A
        # deterministic infinitesimal tie break avoids undefined neighbour
        # radii without changing the raw observations saved alongside it.
        jitter = np.arange(n, dtype=float)[:, None] * 1e-12
        x = outputs + jitter
        y = targets.reshape(-1, 1) + jitter
        if effective_k >= 1:
            joint = np.concatenate((x, y), axis=1)
            radii = cKDTree(joint).query(joint, k=effective_k + 1, p=np.inf)[0][:, -1]
            eps = np.maximum(radii - 1e-15, 0.0)
            x_tree, y_tree = cKDTree(x), cKDTree(y)
            nx = np.asarray([len(x_tree.query_ball_point(point, radius, p=np.inf)) - 1 for point, radius in zip(x, eps)])
            ny = np.asarray([len(y_tree.query_ball_point(point, radius, p=np.inf)) - 1 for point, radius in zip(y, eps)])
            knn_mi = float(digamma(effective_k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
        # ``np.linalg.vector_norm`` is not present on the server's NumPy
        # version; this is the identical Euclidean vector norm.
        summary = np.linalg.norm(outputs, axis=1)
        correlation = spearmanr(summary, targets)
        spearman = float(0.0 if not np.isfinite(correlation.statistic) else correlation.statistic)
    return {
        "pair_count": n,
        "discrete_bins": n_bins,
        "discrete_mi_bits": float(discrete),
        "knn_mi_k": effective_k,
        "knn_mi_bits": float(knn_mi),
        "spearman_rho": float(spearman),
        "information_throughput_mbits_per_s": (
            float(discrete) * 1000.0 / float(wall_clock_seconds) if wall_clock_seconds > 0 else 0.0
        ),
    }


def paired_effect_size(amrl: list[float], fast: list[float]) -> tuple[float, float]:
    """Paired seed-level p-value and Cohen's dz, as required by the manuscript."""
    baseline = np.asarray(amrl, dtype=float)
    optimized = np.asarray(fast, dtype=float)
    if baseline.shape != optimized.shape or len(baseline) < 2:
        return float("nan"), float("nan")
    test = ttest_rel(optimized, baseline, nan_policy="omit")
    delta = optimized - baseline
    std = float(np.std(delta, ddof=1))
    effect = 0.0 if np.isclose(std, 0.0) else float(np.mean(delta) / std)
    return float(test.pvalue), effect


def bootstrap_mean_ci(values: list[float], *, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass
class SeedTrial:
    method: str
    seed: int
    wall_clock_time: float
    # Explicit timing-accounting fields. ``wall_clock_time`` is retained as
    # the backwards-compatible alias for total_lifecycle_wall_clock_time.
    offline_kernel_calibration_admission_time: float
    calibration_time: float
    training_evaluation_wall_clock_time: float
    total_lifecycle_wall_clock_time: float
    final_return: float
    final_return_std: float
    discrete_mi_bits: float
    timing: dict
    stability: dict
    backend: dict
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
