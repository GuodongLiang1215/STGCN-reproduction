from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import PROJECT_ROOT


RUNS = {
    "current graph + current GSO": (
        PROJECT_ROOT / "results" / "paper_full_50epochs" / "final_metrics.csv"
    ),
    "official graph + current GSO": (
        PROJECT_ROOT / "results" / "official_current_gso_50epochs" / "final_metrics.csv"
    ),
    "official graph + original GSO": (
        PROJECT_ROOT
        / "results"
        / "official_original_gso_50epochs"
        / "final_metrics.csv"
    ),
}

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "graph_ablation_comparison"

PAPER_RESULTS = pd.DataFrame(
    {
        "minutes": [15, 30, 45],
        "paper_MAPE_percent": [5.26, 7.33, 8.69],
        "paper_MAE": [2.25, 3.03, 3.57],
        "paper_RMSE": [4.04, 5.70, 6.77],
    }
)


def load_run(run_label: str, metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing result file for '{run_label}':\n" f"{metrics_path}"
        )

    frame = pd.read_csv(metrics_path)

    required_columns = {"horizon_steps", "minutes", "MAPE_percent", "MAE", "RMSE"}

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{metrics_path} is missing columns: " f"{sorted(missing_columns)}"
        )

    frame = frame[["horizon_steps", "minutes", "MAPE_percent", "MAE", "RMSE"]].copy()

    frame["configuration"] = run_label

    return frame


frames = [load_run(run_label, metrics_path) for run_label, metrics_path in RUNS.items()]

long_results = pd.concat(frames, ignore_index=True)

long_results = long_results.sort_values(["minutes", "configuration"]).reset_index(
    drop=True
)


comparison = long_results.merge(PAPER_RESULTS, on="minutes", how="left")

comparison["MAPE_difference_pp"] = (
    comparison["MAPE_percent"] - comparison["paper_MAPE_percent"]
)

comparison["MAE_difference"] = comparison["MAE"] - comparison["paper_MAE"]

comparison["RMSE_difference"] = comparison["RMSE"] - comparison["paper_RMSE"]

comparison["MAPE_abs_difference_pp"] = comparison["MAPE_difference_pp"].abs()

comparison["MAE_abs_difference"] = comparison["MAE_difference"].abs()

comparison["RMSE_abs_difference"] = comparison["RMSE_difference"].abs()


pivot = long_results.pivot(
    index="minutes", columns="configuration", values=["MAPE_percent", "MAE", "RMSE"]
)

current_current = "current graph + current GSO"

official_current = "official graph + current GSO"

official_original = "official graph + original GSO"

pairwise_rows = []

for minutes in (15, 30, 45):
    for metric in ("MAPE_percent", "MAE", "RMSE"):
        baseline_value = float(pivot.loc[minutes, (metric, current_current)])

        official_graph_value = float(pivot.loc[minutes, (metric, official_current)])

        original_gso_value = float(pivot.loc[minutes, (metric, official_original)])

        pairwise_rows.append(
            {
                "minutes": minutes,
                "metric": metric,
                "A_current_graph_current_GSO": baseline_value,
                "B_official_graph_current_GSO": official_graph_value,
                "C_official_graph_original_GSO": original_gso_value,
                "B_minus_A_graph_effect": (official_graph_value - baseline_value),
                "C_minus_B_gso_effect": (original_gso_value - official_graph_value),
                "C_minus_A_total_effect": (original_gso_value - baseline_value),
            }
        )

pairwise_comparison = pd.DataFrame(pairwise_rows)


best_rows = []

for minutes in (15, 30, 45):
    horizon_frame = long_results[long_results["minutes"] == minutes]

    for metric in ("MAPE_percent", "MAE", "RMSE"):
        best_index = horizon_frame[metric].idxmin()

        best_row = long_results.loc[best_index]

        best_rows.append(
            {
                "minutes": minutes,
                "metric": metric,
                "best_configuration": best_row["configuration"],
                "best_value": float(best_row[metric]),
            }
        )

best_configurations = pd.DataFrame(best_rows)


OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

long_path = OUTPUT_DIRECTORY / "graph_ablation_long.csv"

comparison_path = OUTPUT_DIRECTORY / "graph_ablation_vs_paper.csv"

pairwise_path = OUTPUT_DIRECTORY / "pairwise_controlled_effects.csv"

best_path = OUTPUT_DIRECTORY / "best_configurations.csv"

long_results.to_csv(long_path, index=False)

comparison.to_csv(comparison_path, index=False)

pairwise_comparison.to_csv(pairwise_path, index=False)

best_configurations.to_csv(best_path, index=False)


def plot_metric(
    metric_column: str, paper_column: str, ylabel: str, filename: str
) -> None:
    configurations = list(RUNS.keys())

    x_positions = np.arange(3)
    bar_width = 0.22

    plt.figure(figsize=(11, 6))

    for index, configuration in enumerate(configurations):
        values = (
            long_results[long_results["configuration"] == configuration]
            .sort_values("minutes")[metric_column]
            .to_numpy()
        )

        bar_positions = (
            x_positions + (index - (len(configurations) - 1) / 2) * bar_width
        )

        bars = plt.bar(bar_positions, values, bar_width, label=configuration)

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    paper_values = PAPER_RESULTS.sort_values("minutes")[paper_column].to_numpy()

    plt.plot(x_positions, paper_values, marker="o", label="paper STGCN(Cheb)")

    plt.xticks(x_positions, ["15 min", "30 min", "45 min"])

    plt.ylabel(ylabel)
    plt.title(f"STGCN Graph-Pipeline Ablation — {ylabel}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIRECTORY / filename, dpi=200)

    plt.close()


plot_metric(
    metric_column="MAPE_percent",
    paper_column="paper_MAPE_percent",
    ylabel="MAPE (%)",
    filename="graph_ablation_mape.png",
)

plot_metric(
    metric_column="MAE",
    paper_column="paper_MAE",
    ylabel="MAE",
    filename="graph_ablation_mae.png",
)

plot_metric(
    metric_column="RMSE",
    paper_column="paper_RMSE",
    ylabel="RMSE",
    filename="graph_ablation_rmse.png",
)


summary_lines = [
    "STGCN Graph-Pipeline Ablation Summary",
    "=" * 60,
    "",
    "Experiment A:",
    "  current graph + current PyTorch GSO",
    "",
    "Experiment B:",
    "  official STGCN graph + current PyTorch GSO",
    "",
    "Experiment C:",
    "  official STGCN graph + original TensorFlow GSO",
    "",
    "Interpretation of controlled changes:",
    "  B - A isolates the adjacency-matrix effect.",
    "  C - B isolates the GSO-construction effect.",
    "",
]

for minutes in (15, 30, 45):
    summary_lines.append(f"{minutes}-minute horizon")

    horizon_rows = pairwise_comparison[pairwise_comparison["minutes"] == minutes]

    for _, row in horizon_rows.iterrows():
        metric = row["metric"]

        summary_lines.append(f"  {metric}:")

        summary_lines.append(
            "    graph effect (B-A) = " f"{row['B_minus_A_graph_effect']:+.6f}"
        )

        summary_lines.append(
            "    GSO effect (C-B) = " f"{row['C_minus_B_gso_effect']:+.6f}"
        )

        summary_lines.append(
            "    total effect (C-A) = " f"{row['C_minus_A_total_effect']:+.6f}"
        )

    summary_lines.append("")

summary_lines.extend(["Best configuration for each metric:"])

for _, row in best_configurations.iterrows():
    summary_lines.append(
        f"  {int(row['minutes'])}-minute "
        f"{row['metric']}: "
        f"{row['best_configuration']} "
        f"({row['best_value']:.6f})"
    )

summary_lines.extend(
    [
        "",
        "Important caution:",
        (
            "These are single-seed, epoch-50 results. "
            "Small differences between B and C should "
            "not be interpreted as statistically meaningful "
            "until multi-seed experiments are completed."
        ),
    ]
)

summary_path = OUTPUT_DIRECTORY / "graph_ablation_summary.txt"

summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


print("=" * 78)
print("Final test results")
print("=" * 78)

print(
    long_results[["configuration", "minutes", "MAPE_percent", "MAE", "RMSE"]]
    .round(6)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("Controlled effects")
print("=" * 78)

print(pairwise_comparison.round(6).to_string(index=False))

print("\n" + "=" * 78)
print("Best configurations")
print("=" * 78)

print(best_configurations.round(6).to_string(index=False))

print("\nGenerated files:")

for path in (
    long_path,
    comparison_path,
    pairwise_path,
    best_path,
    summary_path,
    OUTPUT_DIRECTORY / "graph_ablation_mape.png",
    OUTPUT_DIRECTORY / "graph_ablation_mae.png",
    OUTPUT_DIRECTORY / "graph_ablation_rmse.png",
):
    print(path)
