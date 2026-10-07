"""Derive manuscript-ready metrics from immutable audited five-seed results.

This script only reads the completed trial JSON files and writes derived tables
to ``results/audited_continuous_r3/derived_metrics``.  It does not alter any
training result, checkpoint, or raw trace.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from amrl_fast.statistics import information_diagnostics  # noqa: E402


ROOT = PROJECT / "results" / "audited_continuous_r3"
OUT = ROOT / "derived_metrics"
ENVS = ("Hopper-v4", "Walker2d-v4")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def bootstrap_ci(values):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(array), size=(N_BOOTSTRAP, len(array)))
    means = array[samples].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_stats(baseline, fast):
    base = np.asarray(baseline, dtype=float)
    opt = np.asarray(fast, dtype=float)
    difference = opt - base
    std = float(difference.std(ddof=1))
    # scipy correctly identifies the test as degenerate when every paired
    # difference is exactly zero, but returns NaN.  The two-sided paired-test
    # limit for an exactly zero observed difference is p=1.0.
    if np.allclose(difference, 0.0):
        return {"paired_ttest_p_value": 1.0, "cohens_dz": 0.0}
    return {
        "paired_ttest_p_value": float(ttest_rel(opt, base, nan_policy="omit").pvalue),
        "cohens_dz": float(difference.mean() / std),
    }


def t90(checkpoints):
    """First recorded checkpoint reaching 90% of the final seed mean return."""
    final = float(checkpoints[-1]["mean_return"])
    threshold = 0.9 * final
    step = next((float(item["step"]) for item in checkpoints if float(item["mean_return"]) >= threshold), None)
    return {"threshold_return": threshold, "t90_step": step}


def aggregate_curve(seed_curves):
    steps = [float(point["step"]) for point in seed_curves[0]]
    means = np.asarray([[float(point["mean_return"]) for point in curve] for curve in seed_curves], dtype=float)
    aggregate = [
        {"step": step, "mean_return": float(means[:, index].mean()), "std_across_seeds": float(means[:, index].std(ddof=1))}
        for index, step in enumerate(steps)
    ]
    threshold = 0.9 * aggregate[-1]["mean_return"]
    first = next((point["step"] for point in aggregate if point["mean_return"] >= threshold), None)
    return aggregate, {"threshold_return": float(threshold), "t90_step": first}


def safe_min(values):
    return float(np.min(np.asarray(values, dtype=float)))


def process_environment(environment: str):
    trials = read_json(ROOT / environment / "seed_trials.json")["paired_seed_trials"]
    per_seed, raw_checkpoints, raw_information = [], [], []
    curves = {"AMRL": [], "AMRL-Fast": []}
    for pair in trials:
        baseline, fast = pair["AMRL"], pair["AMRL_Fast"]
        seed = int(pair["seed"])
        fast_backend = fast["backend"]
        rollout_backend = fast_backend["base_agent_rollout"]
        rollout_runtime = fast_backend["base_agent_rollout_runtime"]
        baseline_raw_info = baseline["audit"]["information_raw_samples"]
        fast_raw_info = fast["audit"]["information_raw_samples"]
        baseline_audit = baseline["audit"]["base_agent_execution"]
        fast_audit = fast["audit"]["base_agent_execution"]
        base_train_eval = float(baseline["training_evaluation_wall_clock_time"])
        fast_train_eval = float(fast["training_evaluation_wall_clock_time"])
        base_lifecycle = float(baseline["total_lifecycle_wall_clock_time"])
        fast_lifecycle = float(fast["total_lifecycle_wall_clock_time"])
        base_meta = sum(float(baseline["timing"][key]) for key in ("meta_learner_forward_time", "meta_learner_backward_time", "hyperparameter_update_time"))
        fast_meta = sum(float(fast["timing"][key]) for key in ("meta_learner_forward_time", "meta_learner_backward_time", "hyperparameter_update_time"))
        # Recompute every estimator from the method's own raw paired samples.
        # This mirrors the paper's fixed k=5 / norm-summary Spearman protocol.
        baseline_info = information_diagnostics(
            baseline_raw_info["controller_outputs"], baseline_raw_info["reward_deltas"], wall_clock_seconds=base_train_eval, k=5
        )
        fast_info = information_diagnostics(
            fast_raw_info["controller_outputs"], fast_raw_info["reward_deltas"], wall_clock_seconds=fast_train_eval, k=5
        )
        saved_baseline_info = baseline["audit"]["information_metrics"]
        saved_fast_info = fast["audit"]["information_metrics"]
        row = {
            "environment": environment,
            "seed": seed,
            "AMRL_final_return": float(baseline["final_return"]),
            "AMRL_Fast_final_return": float(fast["final_return"]),
            "return_difference_fast_minus_amrl": float(fast["final_return"] - baseline["final_return"]),
            "AMRL_train_eval_s": base_train_eval,
            "AMRL_Fast_train_eval_s": fast_train_eval,
            "train_eval_speedup_fraction": float((base_train_eval - fast_train_eval) / base_train_eval),
            "AMRL_lifecycle_s": base_lifecycle,
            "AMRL_Fast_lifecycle_s": fast_lifecycle,
            "lifecycle_speedup_fraction": float((base_lifecycle - fast_lifecycle) / base_lifecycle),
            "AMRL_Fast_calibration_s": float(fast["offline_kernel_calibration_admission_time"]),
            "AMRL_rollout_collection_s": float(baseline_audit["rollout_collection_time"]),
            "AMRL_Fast_rollout_collection_s": float(fast_audit["rollout_collection_time"]),
            "AMRL_ppo_update_s": float(baseline_audit["ppo_update_time"]),
            "AMRL_Fast_ppo_update_s": float(fast_audit["ppo_update_time"]),
            "AMRL_meta_forward_s": float(baseline["timing"]["meta_learner_forward_time"]),
            "AMRL_Fast_meta_forward_s": float(fast["timing"]["meta_learner_forward_time"]),
            "AMRL_meta_backward_s": float(baseline["timing"]["meta_learner_backward_time"]),
            "AMRL_Fast_meta_backward_s": float(fast["timing"]["meta_learner_backward_time"]),
            "AMRL_hyperparameter_application_s": float(baseline["timing"]["hyperparameter_update_time"]),
            "AMRL_Fast_hyperparameter_application_s": float(fast["timing"]["hyperparameter_update_time"]),
            "AMRL_T_meta_s": base_meta,
            "AMRL_Fast_T_meta_s": fast_meta,
            "AMRL_evaluation_s": float(baseline["timing"]["evaluation_time"]),
            "AMRL_Fast_evaluation_s": float(fast["timing"]["evaluation_time"]),
            "AMRL_logging_s": float(baseline["timing"]["logging_time"]),
            "AMRL_Fast_logging_s": float(fast["timing"]["logging_time"]),
            "AMRL_outer_runtime_share": float(base_meta / base_train_eval),
            "AMRL_Fast_outer_runtime_share": float(fast_meta / fast_train_eval),
            "AMRL_outer_runtime_share_lifecycle": float(base_meta / base_lifecycle),
            "AMRL_Fast_outer_runtime_share_lifecycle": float(fast_meta / fast_lifecycle),
            "AMRL_discrete_mi_bits": float(baseline_info["discrete_mi_bits"]),
            "AMRL_Fast_discrete_mi_bits": float(fast_info["discrete_mi_bits"]),
            "AMRL_knn_mi_bits_k5": float(baseline_info["knn_mi_bits"]),
            "AMRL_Fast_knn_mi_bits_k5": float(fast_info["knn_mi_bits"]),
            "AMRL_spearman_rho": float(baseline_info["spearman_rho"]),
            "AMRL_Fast_spearman_rho": float(fast_info["spearman_rho"]),
            "AMRL_information_throughput_mbits_per_s": float(baseline_info["discrete_mi_bits"] * 1000.0 / base_train_eval),
            "AMRL_Fast_information_throughput_mbits_per_s": float(fast_info["discrete_mi_bits"] * 1000.0 / fast_train_eval),
            "information_pair_count_AMRL": int(baseline_info["pair_count"]),
            "information_pair_count_AMRL_Fast": int(fast_info["pair_count"]),
            "native_meta_kernel_speedup": float(fast_backend["measured_speedup"]),
            "rollout_graph_local_speedup": float(rollout_backend["speedup"]),
            "native_meta_eligible_dispatches": int(fast["stability"]["eligible_dispatch_count"]),
            "native_meta_fallback_count": int(fast["stability"]["fallback_count"]),
            "native_meta_fallback_ratio": float(fast["stability"]["fallback_ratio"]),
            "rollout_graph_successful_dispatches": int(rollout_runtime["dispatch_count"]),
            "rollout_graph_fallback_count": int(rollout_runtime["fallback_count"]),
            "nan_inf_count_total": int(fast["stability"]["nan_inf_count"] + rollout_runtime["nan_inf_count"]),
            "meta_output_correlation": float(fast_backend["output_correlation"]),
            "meta_gradient_correlation": float(fast_backend["gradient_correlation"]),
            "rollout_output_correlation": float(rollout_backend["policy_output"]["correlation"]),
            "rollout_value_output_correlation": float(rollout_backend["value_output"]["correlation"]),
            "rollout_input_gradient_correlation": float(rollout_backend["input_gradient"]["correlation"]),
            "rollout_parameter_gradient_correlation": float(rollout_backend["parameter_gradient"]["correlation"]),
            "AMRL_information_recomputed_from_raw_matches_log": bool(all(np.isclose(baseline_info[key], saved_baseline_info[key]) for key in ("discrete_mi_bits", "knn_mi_bits", "spearman_rho"))),
            "AMRL_Fast_information_recomputed_from_raw_matches_log": bool(all(np.isclose(fast_info[key], saved_fast_info[key]) for key in ("discrete_mi_bits", "knn_mi_bits", "spearman_rho"))),
        }
        per_seed.append(row)
        curves["AMRL"].append(baseline["audit"]["checkpoint_evaluations"])
        curves["AMRL-Fast"].append(fast["audit"]["checkpoint_evaluations"])
        for method, trial in (("AMRL", baseline), ("AMRL-Fast", fast)):
            for checkpoint in trial["audit"]["checkpoint_evaluations"]:
                raw_checkpoints.append({
                    "environment": environment,
                    "method": method,
                    "seed": seed,
                    "timestep": checkpoint["step"],
                    "mean_return": checkpoint["mean_return"],
                    "std_return": checkpoint["std_return"],
                    "episode_returns_json": json.dumps(checkpoint["episode_returns"]),
                })
            raw_information.append({
                "environment": environment,
                "method": method,
                "seed": seed,
                "controller_outputs_U": trial["audit"]["information_raw_samples"]["controller_outputs"],
                "reward_deltas_delta_R": trial["audit"]["information_raw_samples"]["reward_deltas"],
                "information_metrics": trial["audit"]["information_metrics"],
            })

    base_returns = [row["AMRL_final_return"] for row in per_seed]
    fast_returns = [row["AMRL_Fast_final_return"] for row in per_seed]
    base_mi = [row["AMRL_discrete_mi_bits"] for row in per_seed]
    fast_mi = [row["AMRL_Fast_discrete_mi_bits"] for row in per_seed]
    base_knn = [row["AMRL_knn_mi_bits_k5"] for row in per_seed]
    fast_knn = [row["AMRL_Fast_knn_mi_bits_k5"] for row in per_seed]
    base_spearman = [row["AMRL_spearman_rho"] for row in per_seed]
    fast_spearman = [row["AMRL_Fast_spearman_rho"] for row in per_seed]
    base_eta = [row["AMRL_information_throughput_mbits_per_s"] for row in per_seed]
    fast_eta = [row["AMRL_Fast_information_throughput_mbits_per_s"] for row in per_seed]
    base_curve, base_t90 = aggregate_curve(curves["AMRL"])
    fast_curve, fast_t90 = aggregate_curve(curves["AMRL-Fast"])
    train_speedups = [row["train_eval_speedup_fraction"] for row in per_seed]
    lifecycle_speedups = [row["lifecycle_speedup_fraction"] for row in per_seed]

    def speedup_summary(values, base_times, fast_times):
        values_array = np.asarray(values, dtype=float)
        return {
            **mean_sd(values),
            "median": float(np.median(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "bootstrap_95_CI": bootstrap_ci(values),
            "ratio_of_means": float((np.mean(base_times) - np.mean(fast_times)) / np.mean(base_times)),
            "aggregation": "mean of paired seed-level speedups (primary); ratio_of_means retained as a diagnostic",
        }

    summary = {
        "environment": environment,
        "n_independent_seeds": len(per_seed),
        "performance": {
            "AMRL_final_return": mean_sd(base_returns),
            "AMRL_Fast_final_return": mean_sd(fast_returns),
            "paired_seed_statistics": paired_stats(base_returns, fast_returns),
            "aggregate_learning_curve_T90": {"AMRL": base_t90, "AMRL_Fast": fast_t90},
            "per_seed_T90": {
                "AMRL": [t90(curve) for curve in curves["AMRL"]],
                "AMRL_Fast": [t90(curve) for curve in curves["AMRL-Fast"]],
            },
        },
        "efficiency": {
            "training_evaluation_speedup": speedup_summary(train_speedups, [row["AMRL_train_eval_s"] for row in per_seed], [row["AMRL_Fast_train_eval_s"] for row in per_seed]),
            "lifecycle_speedup": speedup_summary(lifecycle_speedups, [row["AMRL_lifecycle_s"] for row in per_seed], [row["AMRL_Fast_lifecycle_s"] for row in per_seed]),
            "AMRL_train_eval_s": mean_sd([row["AMRL_train_eval_s"] for row in per_seed]),
            "AMRL_Fast_train_eval_s": mean_sd([row["AMRL_Fast_train_eval_s"] for row in per_seed]),
            "AMRL_lifecycle_s": mean_sd([row["AMRL_lifecycle_s"] for row in per_seed]),
            "AMRL_Fast_lifecycle_s": mean_sd([row["AMRL_Fast_lifecycle_s"] for row in per_seed]),
            "AMRL_Fast_calibration_s": mean_sd([row["AMRL_Fast_calibration_s"] for row in per_seed]),
            "AMRL_outer_runtime_share": {**mean_sd([row["AMRL_outer_runtime_share"] for row in per_seed]), "bootstrap_95_CI": bootstrap_ci([row["AMRL_outer_runtime_share"] for row in per_seed])},
            "AMRL_Fast_outer_runtime_share": {**mean_sd([row["AMRL_Fast_outer_runtime_share"] for row in per_seed]), "bootstrap_95_CI": bootstrap_ci([row["AMRL_Fast_outer_runtime_share"] for row in per_seed])},
            "AMRL_outer_runtime_share_lifecycle": {**mean_sd([row["AMRL_outer_runtime_share_lifecycle"] for row in per_seed]), "bootstrap_95_CI": bootstrap_ci([row["AMRL_outer_runtime_share_lifecycle"] for row in per_seed])},
            "AMRL_Fast_outer_runtime_share_lifecycle": {**mean_sd([row["AMRL_Fast_outer_runtime_share_lifecycle"] for row in per_seed]), "bootstrap_95_CI": bootstrap_ci([row["AMRL_Fast_outer_runtime_share_lifecycle"] for row in per_seed])},
        },
        "information_utility": {
            "discrete_MI_bits": {"AMRL": {**mean_sd(base_mi), "bootstrap_95_CI": bootstrap_ci(base_mi)}, "AMRL_Fast": {**mean_sd(fast_mi), "bootstrap_95_CI": bootstrap_ci(fast_mi)}, "paired_seed_statistics": paired_stats(base_mi, fast_mi)},
            "knn_MI_bits_k5": {"AMRL": {**mean_sd(base_knn), "bootstrap_95_CI": bootstrap_ci(base_knn)}, "AMRL_Fast": {**mean_sd(fast_knn), "bootstrap_95_CI": bootstrap_ci(fast_knn)}, "paired_seed_statistics": paired_stats(base_knn, fast_knn)},
            "spearman_rho": {"AMRL": {**mean_sd(base_spearman), "bootstrap_95_CI": bootstrap_ci(base_spearman)}, "AMRL_Fast": {**mean_sd(fast_spearman), "bootstrap_95_CI": bootstrap_ci(fast_spearman)}, "paired_seed_statistics": paired_stats(base_spearman, fast_spearman)},
            "information_throughput_mbits_per_s": {
                "AMRL": {**mean_sd(base_eta), "bootstrap_95_CI": bootstrap_ci(base_eta)},
                "AMRL_Fast": {**mean_sd(fast_eta), "bootstrap_95_CI": bootstrap_ci(fast_eta)},
                "percentage_increase": float((np.mean(fast_eta) - np.mean(base_eta)) / np.mean(base_eta) * 100.0),
                "wall_clock_scope": "training_evaluation_wall_clock_time; offline calibration excluded",
            },
        },
        "stability": {
            "native_meta_eligible_dispatches_total": int(sum(row["native_meta_eligible_dispatches"] for row in per_seed)),
            "native_meta_successful_dispatches_total": int(sum(row["native_meta_eligible_dispatches"] - row["native_meta_fallback_count"] for row in per_seed)),
            "native_meta_fallback_count_total": int(sum(row["native_meta_fallback_count"] for row in per_seed)),
            "native_meta_fallback_rate_total": float(sum(row["native_meta_fallback_count"] for row in per_seed) / sum(row["native_meta_eligible_dispatches"] for row in per_seed)),
            "rollout_graph_successful_dispatches_total": int(sum(row["rollout_graph_successful_dispatches"] for row in per_seed)),
            "rollout_graph_fallback_count_total": int(sum(row["rollout_graph_fallback_count"] for row in per_seed)),
            "nan_inf_count_total": int(sum(row["nan_inf_count_total"] for row in per_seed)),
            "nan_inf_share_among_fallbacks": (float(sum(row["nan_inf_count_total"] for row in per_seed) / sum(row["native_meta_fallback_count"] + row["rollout_graph_fallback_count"] for row in per_seed)) if sum(row["native_meta_fallback_count"] + row["rollout_graph_fallback_count"] for row in per_seed) else None),
            "minimum_observed_correlations": {
                "meta_output": safe_min([row["meta_output_correlation"] for row in per_seed]),
                "meta_gradient": safe_min([row["meta_gradient_correlation"] for row in per_seed]),
                "rollout_output": safe_min([row["rollout_output_correlation"] for row in per_seed]),
                "rollout_value_output": safe_min([row["rollout_value_output_correlation"] for row in per_seed]),
                "rollout_input_gradient": safe_min([row["rollout_input_gradient_correlation"] for row in per_seed]),
                "rollout_parameter_gradient": safe_min([row["rollout_parameter_gradient_correlation"] for row in per_seed]),
            },
        },
        "component_timing_note": "Saved audit instrumentation records rollout_collection_time (environment interaction plus rollout inference), not an isolated rollout-inference wall-clock; no post hoc separation is inferred.",
    }
    return per_seed, summary, {"AMRL": base_curve, "AMRL_Fast": fast_curve}, raw_checkpoints, raw_information


def fmt(mean, std, digits=2, **_ignored):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, curves, all_checkpoints, all_information = [], {}, {}, [], []
    for env in ENVS:
        rows, summary, learning_curve, checkpoints, information = process_environment(env)
        all_rows.extend(rows)
        summaries[env] = summary
        curves[env] = learning_curve
        all_checkpoints.extend(checkpoints)
        all_information.extend(information)
    write_json(OUT / "paper_metrics_summary.json", summaries)
    write_json(OUT / "learning_curves_and_t90.json", curves)
    write_json(OUT / "information_raw_samples_by_seed.json", all_information)
    with (OUT / "per_seed_raw_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "checkpoint_evaluation_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_checkpoints[0]))
        writer.writeheader()
        writer.writerows(all_checkpoints)

    table_a_rows, table_b_rows = [], []
    for env, item in summaries.items():
        perf, eff, info, stability = item["performance"], item["efficiency"], item["information_utility"], item["stability"]
        table_a_rows.append({
            "Environment": env,
            "AMRL final return": fmt(**perf["AMRL_final_return"]),
            "AMRL-Fast final return": fmt(**perf["AMRL_Fast_final_return"]),
            "paired p": f'{perf["paired_seed_statistics"]["paired_ttest_p_value"]:.4g}',
            "Cohen dz": f'{perf["paired_seed_statistics"]["cohens_dz"]:.3f}',
            "T90 AMRL / Fast (steps)": f'{perf["aggregate_learning_curve_T90"]["AMRL"]["t90_step"]:.0f} / {perf["aggregate_learning_curve_T90"]["AMRL_Fast"]["t90_step"]:.0f}',
            "AMRL train+eval (s)": fmt(**eff["AMRL_train_eval_s"]),
            "Fast train+eval (s)": fmt(**eff["AMRL_Fast_train_eval_s"]),
            "train+eval speedup": fmt(eff["training_evaluation_speedup"]["mean"] * 100, eff["training_evaluation_speedup"]["std"] * 100) + "%",
            "train+eval speedup 95% CI": "[%.2f, %.2f]%%" % tuple(v * 100 for v in eff["training_evaluation_speedup"]["bootstrap_95_CI"]),
            "Fast calibration (s)": fmt(**eff["AMRL_Fast_calibration_s"]),
            "AMRL lifecycle (s)": fmt(**eff["AMRL_lifecycle_s"]),
            "Fast lifecycle (s)": fmt(**eff["AMRL_Fast_lifecycle_s"]),
            "lifecycle speedup": fmt(eff["lifecycle_speedup"]["mean"] * 100, eff["lifecycle_speedup"]["std"] * 100) + "%",
            "rho AMRL / Fast": "%.4f%% / %.4f%%" % (eff["AMRL_outer_runtime_share"]["mean"] * 100, eff["AMRL_Fast_outer_runtime_share"]["mean"] * 100),
        })
        table_b_rows.append({
            "Environment": env,
            "Discrete MI AMRL [95% CI]": fmt(**info["discrete_MI_bits"] ["AMRL"], digits=3) + " [%.3f, %.3f]" % tuple(info["discrete_MI_bits"]["AMRL"]["bootstrap_95_CI"]),
            "Discrete MI Fast [95% CI]": fmt(**info["discrete_MI_bits"]["AMRL_Fast"], digits=3) + " [%.3f, %.3f]" % tuple(info["discrete_MI_bits"]["AMRL_Fast"]["bootstrap_95_CI"]),
            "MI p": f'{info["discrete_MI_bits"]["paired_seed_statistics"]["paired_ttest_p_value"]:.4g}',
            "kNN MI k=5 AMRL / Fast [95% CI]": (
                fmt(**info["knn_MI_bits_k5"]["AMRL"], digits=3)
                + " [%.3f, %.3f] / " % tuple(info["knn_MI_bits_k5"]["AMRL"]["bootstrap_95_CI"])
                + fmt(**info["knn_MI_bits_k5"]["AMRL_Fast"], digits=3)
                + " [%.3f, %.3f]" % tuple(info["knn_MI_bits_k5"]["AMRL_Fast"]["bootstrap_95_CI"])
            ),
            "Spearman AMRL / Fast [95% CI]": (
                fmt(**info["spearman_rho"]["AMRL"], digits=3)
                + " [%.3f, %.3f] / " % tuple(info["spearman_rho"]["AMRL"]["bootstrap_95_CI"])
                + fmt(**info["spearman_rho"]["AMRL_Fast"], digits=3)
                + " [%.3f, %.3f]" % tuple(info["spearman_rho"]["AMRL_Fast"]["bootstrap_95_CI"])
            ),
            "eta AMRL / Fast (mbits/s) [95% CI]": (
                fmt(**info["information_throughput_mbits_per_s"]["AMRL"], digits=3)
                + " [%.3f, %.3f] / " % tuple(info["information_throughput_mbits_per_s"]["AMRL"]["bootstrap_95_CI"])
                + fmt(**info["information_throughput_mbits_per_s"]["AMRL_Fast"], digits=3)
                + " [%.3f, %.3f]" % tuple(info["information_throughput_mbits_per_s"]["AMRL_Fast"]["bootstrap_95_CI"])
            ),
            "eta increase": f'{info["information_throughput_mbits_per_s"]["percentage_increase"]:.2f}%',
            "native eligible / fallback": f'{stability["native_meta_eligible_dispatches_total"]} / {stability["native_meta_fallback_count_total"]} ({stability["native_meta_fallback_rate_total"] * 100:.2f}%)',
            "graph successful dispatches": stability["rollout_graph_successful_dispatches_total"],
            "NaN/Inf": stability["nan_inf_count_total"],
            "min mean-action / value / input-grad / parameter-grad corr": "%.7f / %.7f / %.7f / %.7f" % (stability["minimum_observed_correlations"]["rollout_output"], stability["minimum_observed_correlations"]["rollout_value_output"], stability["minimum_observed_correlations"]["rollout_input_gradient"], stability["minimum_observed_correlations"]["rollout_parameter_gradient"]),
        })
    for name, rows in (("table_A_performance_efficiency", table_a_rows), ("table_B_information_stability", table_b_rows)):
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        headers = list(rows[0])
        md = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in rows:
            md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
        (OUT / f"{name}.md").write_text(md, encoding="utf-8")

    component_rows = []
    components = (
        ("rollout_collection_s", "Rollout collection (environment interaction + inference; not separable post hoc)"),
        ("ppo_update_s", "PPO update"),
        ("meta_forward_s", "Meta forward"),
        ("meta_backward_s", "Meta backward"),
        ("hyperparameter_application_s", "Hyperparameter application"),
        ("evaluation_s", "Evaluation"),
        ("logging_s", "Logging"),
    )
    for env in ENVS:
        env_rows = [row for row in all_rows if row["environment"] == env]
        for key, label in components:
            component_rows.append({
                "Environment": env,
                "Recorded component": label,
                "AMRL seconds (mean ± SD)": fmt(**mean_sd([row[f"AMRL_{key}"] for row in env_rows])),
                "AMRL-Fast seconds (mean ± SD)": fmt(**mean_sd([row[f"AMRL_Fast_{key}"] for row in env_rows])),
            })
    with (OUT / "component_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader(); writer.writerows(component_rows)

    complete_rows = []
    for env, item in summaries.items():
        eff, info, stability = item["efficiency"], item["information_utility"], item["stability"]
        for method, key in (("AMRL", "AMRL"), ("AMRL-Fast", "AMRL_Fast")):
            primary_rho = eff[f"{key}_outer_runtime_share"]
            lifecycle_rho = eff[f"{key}_outer_runtime_share_lifecycle"]
            complete_rows.append({
                "Environment": env,
                "Method": method,
                "Discrete MI (bits), mean ± SD [95% CI]": fmt(**info["discrete_MI_bits"][key], digits=3) + " [%.3f, %.3f]" % tuple(info["discrete_MI_bits"][key]["bootstrap_95_CI"]),
                "kNN MI (k=5), mean ± SD [95% CI]": fmt(**info["knn_MI_bits_k5"][key], digits=3) + " [%.3f, %.3f]" % tuple(info["knn_MI_bits_k5"][key]["bootstrap_95_CI"]),
                "Spearman rho, mean ± SD [95% CI]": fmt(**info["spearman_rho"][key], digits=3) + " [%.3f, %.3f]" % tuple(info["spearman_rho"][key]["bootstrap_95_CI"]),
                "eta_info (mbits/s), mean ± SD [95% CI]": fmt(**info["information_throughput_mbits_per_s"][key], digits=3) + " [%.3f, %.3f]" % tuple(info["information_throughput_mbits_per_s"][key]["bootstrap_95_CI"]),
                "paired MI p vs AMRL": "--" if method == "AMRL" else f'{info["discrete_MI_bits"]["paired_seed_statistics"]["paired_ttest_p_value"]:.4g}',
                "paired kNN p vs AMRL": "--" if method == "AMRL" else f'{info["knn_MI_bits_k5"]["paired_seed_statistics"]["paired_ttest_p_value"]:.4g}',
                "paired Spearman p vs AMRL": "--" if method == "AMRL" else f'{info["spearman_rho"]["paired_seed_statistics"]["paired_ttest_p_value"]:.4g}',
                "train+eval speedup, mean ± SD [95% CI]": "--" if method == "AMRL" else fmt(eff["training_evaluation_speedup"]["mean"] * 100, eff["training_evaluation_speedup"]["std"] * 100) + "%% [%.2f, %.2f]%%" % tuple(v * 100 for v in eff["training_evaluation_speedup"]["bootstrap_95_CI"]),
                "lifecycle speedup, mean ± SD [95% CI]": "--" if method == "AMRL" else fmt(eff["lifecycle_speedup"]["mean"] * 100, eff["lifecycle_speedup"]["std"] * 100) + "%% [%.2f, %.2f]%%" % tuple(v * 100 for v in eff["lifecycle_speedup"]["bootstrap_95_CI"]),
                "outer-loop rho train+eval, mean ± SD [95% CI]": fmt(primary_rho["mean"] * 100, primary_rho["std"] * 100, digits=4) + "%% [%.4f, %.4f]%%" % tuple(v * 100 for v in primary_rho["bootstrap_95_CI"]),
                "outer-loop rho lifecycle, mean ± SD [95% CI]": fmt(lifecycle_rho["mean"] * 100, lifecycle_rho["std"] * 100, digits=4) + "%% [%.4f, %.4f]%%" % tuple(v * 100 for v in lifecycle_rho["bootstrap_95_CI"]),
                "fallback rate": "--" if method == "AMRL" else f'{stability["native_meta_fallback_rate_total"] * 100:.2f}%',
                "NaN/Inf": "--" if method == "AMRL" else stability["nan_inf_count_total"],
            })
    for name, rows in (("complete_statistics_summary", complete_rows),):
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        headers = list(rows[0])
        md = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in rows:
            md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
        (OUT / f"{name}.md").write_text(md, encoding="utf-8")

    speedup_rows = []
    for env, item in summaries.items():
        for scope, key in (("training+evaluation", "training_evaluation_speedup"), ("lifecycle", "lifecycle_speedup")):
            stat = item["efficiency"][key]
            speedup_rows.append({
                "environment": env,
                "time_scope": scope,
                "primary_aggregation": stat["aggregation"],
                "mean_of_paired_seed_speedups_fraction": stat["mean"],
                "std": stat["std"],
                "median": stat["median"],
                "min": stat["min"],
                "max": stat["max"],
                "bootstrap_95_CI": json.dumps(stat["bootstrap_95_CI"]),
                "ratio_of_mean_times_fraction": stat["ratio_of_means"],
            })
    with (OUT / "speedup_aggregation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speedup_rows[0]))
        writer.writeheader(); writer.writerows(speedup_rows)


if __name__ == "__main__":
    main()
