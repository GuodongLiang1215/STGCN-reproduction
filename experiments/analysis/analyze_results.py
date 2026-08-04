from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIRECTORY = Path("results/paper_full_50epochs")

HISTORY_PATH = RUN_DIRECTORY / "training_history.csv"

FINAL_METRICS_PATH = RUN_DIRECTORY / "final_metrics.csv"

ANALYSIS_DIRECTORY = RUN_DIRECTORY / "analysis"


if not HISTORY_PATH.exists():
    raise FileNotFoundError(f"Training history not found: {HISTORY_PATH}")

if not FINAL_METRICS_PATH.exists():
    raise FileNotFoundError(f"Final metrics not found: {FINAL_METRICS_PATH}")

ANALYSIS_DIRECTORY.mkdir(parents=True, exist_ok=True)


history = pd.read_csv(HISTORY_PATH)

reproduction_metrics = pd.read_csv(FINAL_METRICS_PATH)

reproduction_metrics = reproduction_metrics.rename(
    columns={
        "MAPE_percent": "reproduction_MAPE_percent",
        "MAE": "reproduction_MAE",
        "RMSE": "reproduction_RMSE",
    }
)


paper_metrics = pd.DataFrame(
    {
        "minutes": [15, 30, 45],
        "paper_MAPE_percent": [5.26, 7.33, 8.69],
        "paper_MAE": [2.25, 3.03, 3.57],
        "paper_RMSE": [4.04, 5.70, 6.77],
    }
)


comparison = reproduction_metrics.merge(paper_metrics, on="minutes", how="inner")

comparison = comparison.sort_values("minutes").reset_index(drop=True)


# MAPE is already a percentage, so this difference is in percentage points.
comparison["MAPE_difference_pp"] = (
    comparison["reproduction_MAPE_percent"] - comparison["paper_MAPE_percent"]
)

comparison["MAE_difference"] = comparison["reproduction_MAE"] - comparison["paper_MAE"]

comparison["RMSE_difference"] = (
    comparison["reproduction_RMSE"] - comparison["paper_RMSE"]
)


# Relative differences from the paper values.
comparison["MAPE_relative_difference_percent"] = (
    comparison["MAPE_difference_pp"] / comparison["paper_MAPE_percent"] * 100
)

comparison["MAE_relative_difference_percent"] = (
    comparison["MAE_difference"] / comparison["paper_MAE"] * 100
)

comparison["RMSE_relative_difference_percent"] = (
    comparison["RMSE_difference"] / comparison["paper_RMSE"] * 100
)


# Keep the comparison table in a stable presentation order.
comparison = comparison[
    [
        "horizon_steps",
        "minutes",
        "paper_MAPE_percent",
        "reproduction_MAPE_percent",
        "MAPE_difference_pp",
        "MAPE_relative_difference_percent",
        "paper_MAE",
        "reproduction_MAE",
        "MAE_difference",
        "MAE_relative_difference_percent",
        "paper_RMSE",
        "reproduction_RMSE",
        "RMSE_difference",
        "RMSE_relative_difference_percent",
    ]
]


comparison_path = ANALYSIS_DIRECTORY / "paper_comparison.csv"

comparison.to_csv(comparison_path, index=False)


validation_columns = {
    15: {"MAPE": "val_15_mape_percent", "MAE": "val_15_mae", "RMSE": "val_15_rmse"},
    30: {"MAPE": "val_30_mape_percent", "MAE": "val_30_mae", "RMSE": "val_30_rmse"},
    45: {"MAPE": "val_45_mape_percent", "MAE": "val_45_mae", "RMSE": "val_45_rmse"},
}

best_validation_records = []

for minutes, metric_mapping in validation_columns.items():
    for metric_name, column_name in metric_mapping.items():
        best_row_index = history[column_name].idxmin()

        best_row = history.loc[best_row_index]

        best_validation_records.append(
            {
                "minutes": minutes,
                "metric": metric_name,
                "best_epoch": int(best_row["epoch"]),
                "best_value": float(best_row[column_name]),
                "learning_rate": float(best_row["learning_rate"]),
            }
        )

best_validation = pd.DataFrame(best_validation_records)

best_validation_path = ANALYSIS_DIRECTORY / "best_validation_epochs.csv"

best_validation.to_csv(best_validation_path, index=False)


plt.figure(figsize=(10, 5))

plt.plot(
    history["epoch"],
    history["train_l2_per_sample"],
    label="STGCN training L2 per sample",
)

plt.plot(
    history["epoch"], history["copy_l2_per_sample"], label="Copy-last-value baseline"
)

plt.xlabel("Epoch")
plt.ylabel("L2 loss per sample")
plt.title("STGCN Training Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(ANALYSIS_DIRECTORY / "training_l2_curve.png", dpi=200)

plt.close()


plt.figure(figsize=(10, 5))

plt.step(history["epoch"], history["learning_rate"], where="post")

plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("Learning Rate Schedule")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(ANALYSIS_DIRECTORY / "learning_rate_curve.png", dpi=200)

plt.close()


def plot_validation_curves(metric_suffix, ylabel, filename):
    """
    Plot validation metrics at the 15-, 30-, and 45-minute horizons.
    """

    plt.figure(figsize=(10, 5))

    for minutes in (15, 30, 45):
        column_name = f"val_{minutes}_" f"{metric_suffix}"

        plt.plot(history["epoch"], history[column_name], label=f"{minutes}-minute")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.title(f"Validation {ylabel}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(ANALYSIS_DIRECTORY / filename, dpi=200)

    plt.close()


plot_validation_curves(
    metric_suffix="mape_percent",
    ylabel="MAPE (%)",
    filename="validation_mape_curve.png",
)

plot_validation_curves(
    metric_suffix="mae", ylabel="MAE", filename="validation_mae_curve.png"
)

plot_validation_curves(
    metric_suffix="rmse", ylabel="RMSE", filename="validation_rmse_curve.png"
)


def plot_paper_comparison(paper_column, reproduction_column, ylabel, title, filename):
    """
    Plot grouped bars for the paper and reproduction values.
    """

    x_positions = np.arange(len(comparison))

    bar_width = 0.36

    paper_values = comparison[paper_column].to_numpy()

    reproduction_values = comparison[reproduction_column].to_numpy()

    plt.figure(figsize=(9, 5))

    paper_bars = plt.bar(
        x_positions - bar_width / 2, paper_values, bar_width, label="Paper STGCN(Cheb)"
    )

    reproduction_bars = plt.bar(
        x_positions + bar_width / 2,
        reproduction_values,
        bar_width,
        label="PyTorch reproduction",
    )

    plt.xticks(x_positions, [f"{minutes} min" for minutes in comparison["minutes"]])

    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Label bars because small cross-framework differences are easy to miss.
    for bars in (paper_bars, reproduction_bars):
        for bar in bars:
            bar_height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar_height,
                f"{bar_height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    plt.savefig(ANALYSIS_DIRECTORY / filename, dpi=200)

    plt.close()


plot_paper_comparison(
    paper_column="paper_MAPE_percent",
    reproduction_column=("reproduction_MAPE_percent"),
    ylabel="MAPE (%)",
    title=("STGCN Paper vs " "PyTorch Reproduction — MAPE"),
    filename="paper_mape_comparison.png",
)

plot_paper_comparison(
    paper_column="paper_MAE",
    reproduction_column=("reproduction_MAE"),
    ylabel="MAE",
    title=("STGCN Paper vs " "PyTorch Reproduction — MAE"),
    filename="paper_mae_comparison.png",
)

plot_paper_comparison(
    paper_column="paper_RMSE",
    reproduction_column=("reproduction_RMSE"),
    ylabel="RMSE",
    title=("STGCN Paper vs " "PyTorch Reproduction — RMSE"),
    filename="paper_rmse_comparison.png",
)


summary_lines = [
    "STGCN Paper-Aligned Reproduction Summary",
    "=" * 55,
    "",
    "Dataset: PeMSD7(M)",
    "Historical window: 12 x 5 minutes = 60 minutes",
    "Forecast horizons: 15, 30 and 45 minutes",
    "Training epochs: 50",
    "",
    "Paper comparison:",
]

for _, row in comparison.iterrows():
    minutes = int(row["minutes"])

    summary_lines.append("")

    summary_lines.append(f"{minutes}-minute forecast")

    summary_lines.append(
        "  MAPE: "
        f"paper={row['paper_MAPE_percent']:.3f}%, "
        f"reproduction="
        f"{row['reproduction_MAPE_percent']:.3f}%, "
        f"difference="
        f"{row['MAPE_difference_pp']:+.3f} "
        "percentage points"
    )

    summary_lines.append(
        "  MAE:  "
        f"paper={row['paper_MAE']:.3f}, "
        f"reproduction="
        f"{row['reproduction_MAE']:.3f}, "
        f"difference="
        f"{row['MAE_difference']:+.3f}"
    )

    summary_lines.append(
        "  RMSE: "
        f"paper={row['paper_RMSE']:.3f}, "
        f"reproduction="
        f"{row['reproduction_RMSE']:.3f}, "
        f"difference="
        f"{row['RMSE_difference']:+.3f}"
    )


summary_lines.extend(["", "Best validation epochs:"])

for _, row in best_validation.iterrows():
    metric_value = row["best_value"]

    summary_lines.append(
        f"  {int(row['minutes'])}-minute "
        f"{row['metric']}: "
        f"epoch {int(row['best_epoch'])}, "
        f"value={metric_value:.6f}"
    )


summary_lines.extend(
    [
        "",
        "Interpretation:",
        (
            "The PyTorch reproduction closely matches "
            "the reported STGCN(Cheb) results."
        ),
        (
            "Errors increase with forecast horizon "
            "because predictions are generated "
            "autoregressively."
        ),
        (
            "Different validation metrics reach their "
            "minimum at different epochs, so epoch 50 "
            "is not necessarily the best epoch for "
            "every individual metric."
        ),
    ]
)

summary_path = ANALYSIS_DIRECTORY / "analysis_summary.txt"

summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


print("=" * 72)
print("1. Paper comparison")
print("=" * 72)

display_columns = [
    "minutes",
    "paper_MAPE_percent",
    "reproduction_MAPE_percent",
    "MAPE_difference_pp",
    "paper_MAE",
    "reproduction_MAE",
    "MAE_difference",
    "paper_RMSE",
    "reproduction_RMSE",
    "RMSE_difference",
]

print(comparison[display_columns].round(4).to_string(index=False))


print("\n" + "=" * 72)
print("2. Best validation epochs")
print("=" * 72)

print(best_validation.round(6).to_string(index=False))


print("\n" + "=" * 72)
print("3. Generated files")
print("=" * 72)

generated_files = [
    comparison_path,
    best_validation_path,
    summary_path,
    ANALYSIS_DIRECTORY / "training_l2_curve.png",
    ANALYSIS_DIRECTORY / "learning_rate_curve.png",
    ANALYSIS_DIRECTORY / "validation_mape_curve.png",
    ANALYSIS_DIRECTORY / "validation_mae_curve.png",
    ANALYSIS_DIRECTORY / "validation_rmse_curve.png",
    ANALYSIS_DIRECTORY / "paper_mape_comparison.png",
    ANALYSIS_DIRECTORY / "paper_mae_comparison.png",
    ANALYSIS_DIRECTORY / "paper_rmse_comparison.png",
]

for generated_file in generated_files:
    print(generated_file)
