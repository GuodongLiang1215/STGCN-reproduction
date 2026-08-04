from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import PROJECT_ROOT


SEED_FILES = {
    42: (
        PROJECT_ROOT
        / "results"
        / "official_original_gso_50epochs"
        / "final_metrics.csv"
    ),
    123: (
        PROJECT_ROOT
        / "results"
        / "official_original_gso_seed123_protocol"
        / "selection_protocol"
        / "selected_checkpoint_epoch_049_metrics.csv"
    ),
    2026: (
        PROJECT_ROOT
        / "results"
        / "official_original_gso_seed2026_protocol"
        / "selection_protocol"
        / "selected_checkpoint_epoch_048_metrics.csv"
    ),
}

SELECTED_EPOCHS = {42: 50, 123: 49, 2026: 48}

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "multiseed_summary"

PAPER_RESULTS = pd.DataFrame(
    {
        "minutes": [15, 30, 45],
        "paper_MAPE_percent": [5.26, 7.33, 8.69],
        "paper_MAE": [2.25, 3.03, 3.57],
        "paper_RMSE": [4.04, 5.70, 6.77],
    }
)


frames = []

for seed, metrics_path in SEED_FILES.items():
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file for seed {seed}:\n" f"{metrics_path}"
        )

    frame = pd.read_csv(metrics_path)

    required_columns = {"horizon_steps", "minutes", "MAPE_percent", "MAE", "RMSE"}

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{metrics_path} is missing columns: " f"{sorted(missing_columns)}"
        )

    frame = frame[["horizon_steps", "minutes", "MAPE_percent", "MAE", "RMSE"]].copy()

    frame["seed"] = seed
    frame["selected_epoch"] = SELECTED_EPOCHS[seed]
    frame["metrics_path"] = str(metrics_path.relative_to(PROJECT_ROOT))

    frames.append(frame)

per_seed = pd.concat(frames, ignore_index=True)

per_seed = (
    per_seed[
        [
            "seed",
            "selected_epoch",
            "horizon_steps",
            "minutes",
            "MAPE_percent",
            "MAE",
            "RMSE",
            "metrics_path",
        ]
    ]
    .sort_values(["minutes", "seed"])
    .reset_index(drop=True)
)


summary_rows = []

for minutes in (15, 30, 45):
    horizon_frame = per_seed[per_seed["minutes"] == minutes]

    row = {"minutes": minutes, "n_seeds": len(horizon_frame)}

    for metric in ("MAPE_percent", "MAE", "RMSE"):
        values = horizon_frame[metric].to_numpy(dtype=np.float64)

        row[f"{metric}_mean"] = float(np.mean(values))

        # Sample standard deviation: ddof=1
        row[f"{metric}_std"] = float(np.std(values, ddof=1))

        row[f"{metric}_min"] = float(np.min(values))

        row[f"{metric}_max"] = float(np.max(values))

    summary_rows.append(row)

mean_std = pd.DataFrame(summary_rows)


comparison = mean_std.merge(PAPER_RESULTS, on="minutes", how="left")

comparison["MAPE_mean_minus_paper_pp"] = (
    comparison["MAPE_percent_mean"] - comparison["paper_MAPE_percent"]
)

comparison["MAE_mean_minus_paper"] = comparison["MAE_mean"] - comparison["paper_MAE"]

comparison["RMSE_mean_minus_paper"] = comparison["RMSE_mean"] - comparison["paper_RMSE"]


OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

per_seed_path = OUTPUT_DIRECTORY / "per_seed_metrics.csv"

mean_std_path = OUTPUT_DIRECTORY / "mean_std_metrics.csv"

comparison_path = OUTPUT_DIRECTORY / "mean_std_vs_paper.csv"

per_seed.to_csv(per_seed_path, index=False)

mean_std.to_csv(mean_std_path, index=False)

comparison.to_csv(comparison_path, index=False)


def plot_metric(metric: str, paper_column: str, ylabel: str, filename: str):
    x = np.arange(3)

    means = mean_std[f"{metric}_mean"].to_numpy()

    stds = mean_std[f"{metric}_std"].to_numpy()

    paper_values = PAPER_RESULTS[paper_column].to_numpy()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        x, means, yerr=stds, capsize=6, label="PyTorch reproduction (mean ± SD)"
    )

    plt.plot(x, paper_values, marker="o", label="Paper STGCN(Cheb)")

    plt.xticks(x, ["15 min", "30 min", "45 min"])

    plt.ylabel(ylabel)
    plt.title(f"STGCN Three-Seed Reproduction — {ylabel}")

    plt.grid(axis="y", alpha=0.3)

    plt.legend()

    for bar, mean, std in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3f} ± {std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(OUTPUT_DIRECTORY / filename, dpi=200)

    plt.close()


plot_metric(
    metric="MAPE_percent",
    paper_column="paper_MAPE_percent",
    ylabel="MAPE (%)",
    filename="multiseed_mape.png",
)

plot_metric(
    metric="MAE", paper_column="paper_MAE", ylabel="MAE", filename="multiseed_mae.png"
)

plot_metric(
    metric="RMSE",
    paper_column="paper_RMSE",
    ylabel="RMSE",
    filename="multiseed_rmse.png",
)


summary_lines = [
    "STGCN Three-Seed Reproduction Summary",
    "=" * 58,
    "",
    "Configuration:",
    "  official PeMSD7(M) adjacency matrix",
    "  original TensorFlow STGCN scaled Laplacian",
    "  paper-aligned PyTorch architecture",
    "  original validation-trigger selection protocol",
    "",
    "Selected checkpoints:",
]

for seed in sorted(SELECTED_EPOCHS):
    summary_lines.append(f"  seed {seed}: " f"epoch {SELECTED_EPOCHS[seed]}")

summary_lines.extend(["", "Mean ± sample standard deviation:"])

for _, row in mean_std.iterrows():
    minutes = int(row["minutes"])

    summary_lines.extend(
        [
            "",
            f"{minutes}-minute horizon",
            (
                "  MAPE: "
                f"{row['MAPE_percent_mean']:.6f}% "
                f"± {row['MAPE_percent_std']:.6f}%"
            ),
            ("  MAE:  " f"{row['MAE_mean']:.6f} " f"± {row['MAE_std']:.6f}"),
            ("  RMSE: " f"{row['RMSE_mean']:.6f} " f"± {row['RMSE_std']:.6f}"),
        ]
    )

summary_lines.extend(["", "Mean result versus paper:"])

for _, row in comparison.iterrows():
    minutes = int(row["minutes"])

    summary_lines.extend(
        [
            "",
            f"{minutes}-minute horizon",
            (
                "  MAPE difference: "
                f"{row['MAPE_mean_minus_paper_pp']:+.6f} "
                "percentage points"
            ),
            ("  MAE difference: " f"{row['MAE_mean_minus_paper']:+.6f}"),
            ("  RMSE difference: " f"{row['RMSE_mean_minus_paper']:+.6f}"),
        ]
    )

summary_lines.extend(
    [
        "",
        "Interpretation:",
        (
            "The standard deviation increases with the forecast "
            "horizon, which is consistent with autoregressive "
            "error accumulation."
        ),
        (
            "Three seeds provide a first stability check, but they "
            "are still a small sample and should not be treated as "
            "a complete statistical study."
        ),
    ]
)

summary_path = OUTPUT_DIRECTORY / "multiseed_summary.txt"

summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


print("=" * 78)
print("Per-seed metrics")
print("=" * 78)

print(
    per_seed[["seed", "selected_epoch", "minutes", "MAPE_percent", "MAE", "RMSE"]]
    .round(6)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("Mean ± sample standard deviation")
print("=" * 78)

display_columns = [
    "minutes",
    "n_seeds",
    "MAPE_percent_mean",
    "MAPE_percent_std",
    "MAE_mean",
    "MAE_std",
    "RMSE_mean",
    "RMSE_std",
]

print(mean_std[display_columns].round(6).to_string(index=False))

print("\n" + "=" * 78)
print("Mean result versus paper")
print("=" * 78)

print(
    comparison[
        [
            "minutes",
            "MAPE_mean_minus_paper_pp",
            "MAE_mean_minus_paper",
            "RMSE_mean_minus_paper",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print("\nGenerated files:")

for path in (
    per_seed_path,
    mean_std_path,
    comparison_path,
    summary_path,
    OUTPUT_DIRECTORY / "multiseed_mape.png",
    OUTPUT_DIRECTORY / "multiseed_mae.png",
    OUTPUT_DIRECTORY / "multiseed_rmse.png",
):
    print(path)
