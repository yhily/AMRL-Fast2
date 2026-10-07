"""Read-only audit of KSG MI and Walker paired tests from completed trial logs.

No training result is rewritten.  The output is a transparent calculation
record for the finite-sample kNN estimator and independent paired t tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import spearmanr, ttest_rel


ROOT = Path(__file__).resolve().parents[1] / "results" / "audited_continuous_r3"
OUT = ROOT / "derived_metrics" / "knn_pvalue_audit.json"


def ksg_audit(controller_outputs, reward_deltas, requested_k=5):
    """Independent transcription of the project KSG estimator, with traces."""
    x_raw = np.asarray(controller_outputs, dtype=float)
    y_raw = np.asarray(reward_deltas, dtype=float).reshape(-1, 1)
    n = int(y_raw.shape[0])
    k = min(int(requested_k), max(n - 1, 0))
    jitter = np.arange(n, dtype=float)[:, None] * 1e-12
    x, y = x_raw + jitter, y_raw + jitter
    joint = np.concatenate((x, y), axis=1)
    radii = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, -1]
    eps = np.maximum(radii - 1e-15, 0.0)
    x_tree, y_tree = cKDTree(x), cKDTree(y)
    nx = np.asarray([len(x_tree.query_ball_point(point, radius, p=np.inf)) - 1 for point, radius in zip(x, eps)])
    ny = np.asarray([len(y_tree.query_ball_point(point, radius, p=np.inf)) - 1 for point, radius in zip(y, eps)])
    terms = digamma(nx + 1) + digamma(ny + 1)
    estimate = float(digamma(k) + digamma(n) - np.mean(terms))
    summary_norm = np.linalg.norm(x_raw, axis=1)
    rho = spearmanr(summary_norm, y_raw.reshape(-1)).statistic
    return {
        "formula": "psi(k)+psi(n)-mean_i[psi(n_x,i+1)+psi(n_y,i+1)]",
        "clipping_or_floor_applied": False,
        "finite_sample_bias_correction": "No separate correction; this is the saved KSG-style digamma estimator.",
        "requested_k": int(requested_k),
        "effective_k": int(k),
        "U_shape": list(x_raw.shape),
        "delta_R_shape": list(y_raw.shape),
        "effective_samples": n,
        "joint_shape": list(joint.shape),
        "joint_metric": "Chebyshev (p=infinity)",
        "tie_handling": "deterministic row-index jitter of 1e-12 before neighbour search",
        "radius": radii.tolist(),
        "epsilon": eps.tolist(),
        "n_x": nx.tolist(),
        "n_y": ny.tolist(),
        "mean_digamma_count_term": float(np.mean(terms)),
        "digamma_k_plus_digamma_n": float(digamma(k) + digamma(n)),
        "knn_mi_bits_as_stored": estimate,
        "spearman_input": "Euclidean norm of the original three-dimensional U, as in the project paper protocol",
        "spearman_rho_as_stored": float(rho),
    }


def main():
    report = {
        "scope": "Read-only recomputation from each method's own raw U and delta-R samples.",
        "kNN_unit_note": "The digamma KSG formula uses natural-log units (nats). The legacy field name knn_mi_bits is therefore a naming/unit-label issue, not a numerical transformation: no division by ln(2), clipping, flooring, or bias correction is applied.",
        "environments": {},
    }
    for env in ("Hopper-v4", "Walker2d-v4"):
        pairs = json.loads((ROOT / env / "seed_trials.json").read_text(encoding="utf-8"))["paired_seed_trials"]
        env_report = {"seeds": [], "walker_paired_tests": None}
        values = {"discrete": {"AMRL": [], "AMRL_Fast": []}, "knn": {"AMRL": [], "AMRL_Fast": []}, "spearman": {"AMRL": [], "AMRL_Fast": []}}
        for pair in pairs:
            seed_report = {"seed": int(pair["seed"]), "methods": {}}
            for label, trial_key in (("AMRL", "AMRL"), ("AMRL_Fast", "AMRL_Fast")):
                trial = pair[trial_key]
                raw = trial["audit"]["information_raw_samples"]
                saved = trial["audit"]["information_metrics"]
                audit = ksg_audit(raw["controller_outputs"], raw["reward_deltas"], requested_k=5)
                audit["saved_knn_mi_bits"] = float(saved["knn_mi_bits"])
                audit["saved_spearman_rho"] = float(saved["spearman_rho"])
                audit["saved_discrete_mi_bits"] = float(saved["discrete_mi_bits"])
                audit["knn_matches_saved"] = bool(np.isclose(audit["knn_mi_bits_as_stored"], saved["knn_mi_bits"]))
                audit["spearman_matches_saved"] = bool(np.isclose(audit["spearman_rho_as_stored"], saved["spearman_rho"]))
                seed_report["methods"][label] = audit
                values["discrete"][label].append(float(saved["discrete_mi_bits"]))
                values["knn"][label].append(float(audit["knn_mi_bits_as_stored"]))
                values["spearman"][label].append(float(audit["spearman_rho_as_stored"]))
            env_report["seeds"].append(seed_report)
        if env == "Walker2d-v4":
            tests = {}
            for name in ("discrete", "knn", "spearman"):
                base = np.asarray(values[name]["AMRL"], dtype=float)
                fast = np.asarray(values[name]["AMRL_Fast"], dtype=float)
                # A separate ttest_rel call is deliberately made for every metric.
                result = ttest_rel(fast, base, nan_policy="omit")
                tests[name] = {
                    "AMRL_values": base.tolist(),
                    "AMRL_Fast_values": fast.tolist(),
                    "paired_difference_fast_minus_amrl": (fast - base).tolist(),
                    "paired_t_statistic": float(result.statistic),
                    "paired_ttest_two_sided_p_value": float(result.pvalue),
                    "test_called_independently": True,
                }
            env_report["walker_paired_tests"] = tests
        report["environments"][env] = env_report
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Read-only KSG MI and paired-test audit",
        "",
        "- Formula: `psi(k) + psi(n) - mean[psi(n_x+1) + psi(n_y+1)]`.",
        "- No clipping, floor, normalization, or extra bias correction is applied.",
        "- The formula's native unit is nats; `knn_mi_bits` is a legacy field name. No numerical correction is applied because the requested protocol is to retain the existing estimator.",
        "- Each run has `n=9`, `U.shape=(9,3)`, `delta-R.shape=(9,1)`, and effective `k=5`.",
        "",
        "## Per-seed kNN estimates",
        "",
        "| Environment | Seed | AMRL kNN | Fast kNN | AMRL mean count term | Fast mean count term | Recomputed values match saved logs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for env, env_report in report["environments"].items():
        for item in env_report["seeds"]:
            base, fast = item["methods"]["AMRL"], item["methods"]["AMRL_Fast"]
            matched = base["knn_matches_saved"] and fast["knn_matches_saved"]
            lines.append(
                f"| {env} | {item['seed']} | {base['knn_mi_bits_as_stored']:.12f} | {fast['knn_mi_bits_as_stored']:.12f} | "
                f"{base['mean_digamma_count_term']:.12f} | {fast['mean_digamma_count_term']:.12f} | {matched} |"
            )
    lines.extend([
        "",
        "## Walker2d-v4: independent paired tests",
        "",
        "| Metric | AMRL (five seeds) | Fast (five seeds) | Fast − AMRL | t | two-sided paired p |",
        "|---|---|---|---|---:|---:|",
    ])
    for name, values in report["environments"]["Walker2d-v4"]["walker_paired_tests"].items():
        fmt = lambda array: ", ".join(f"{value:.12f}" for value in array)
        lines.append(
            f"| {name} | {fmt(values['AMRL_values'])} | {fmt(values['AMRL_Fast_values'])} | "
            f"{fmt(values['paired_difference_fast_minus_amrl'])} | {values['paired_t_statistic']:.6f} | {values['paired_ttest_two_sided_p_value']:.12f} |"
        )
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
