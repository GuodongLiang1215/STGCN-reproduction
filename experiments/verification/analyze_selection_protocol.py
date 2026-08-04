from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import PROJECT_ROOT


def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the original STGCN validation-selection "
            "logic using an existing PyTorch training history."
        )
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="official_original_gso_50epochs",
        help=("Run directory under results/ and checkpoints/."),
    )

    return parser.parse_args()


METRICS = [
    ("15_min_MAPE", "val_15_mape_percent", 0.01),
    ("15_min_MAE", "val_15_mae", 1.0),
    ("15_min_RMSE", "val_15_rmse", 1.0),
    ("30_min_MAPE", "val_30_mape_percent", 0.01),
    ("30_min_MAE", "val_30_mae", 1.0),
    ("30_min_RMSE", "val_30_rmse", 1.0),
    ("45_min_MAPE", "val_45_mape_percent", 0.01),
    ("45_min_MAE", "val_45_mae", 1.0),
    ("45_min_RMSE", "val_45_rmse", 1.0),
]


def build_validation_vector(row: pd.Series) -> np.ndarray:
    """
    Match the original TensorFlow metric vector:

    [15 MAPE, 15 MAE, 15 RMSE,
     30 MAPE, 30 MAE, 30 RMSE,
     45 MAPE, 45 MAE, 45 RMSE]

    MAPE is converted from percent to fraction.
    """

    values = []

    for _, column_name, scale in METRICS:
        values.append(float(row[column_name]) * scale)

    return np.asarray(values, dtype=np.float64)


def checkpoint_path_for_epoch(checkpoint_directory: Path, epoch: int) -> Path:
    return checkpoint_directory / f"paper_stgcn_epoch_{epoch:03d}.pt"


def main():
    args = get_arguments()

    project_root = PROJECT_ROOT

    result_directory = project_root / "results" / args.run_name

    checkpoint_directory = project_root / "checkpoints" / args.run_name

    history_path = result_directory / "training_history.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found:\n{history_path}")

    history = pd.read_csv(history_path).sort_values("epoch").reset_index(drop=True)

    required_columns = {"epoch", *[column_name for _, column_name, _ in METRICS]}

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            "Missing training-history columns: " f"{sorted(missing_columns)}"
        )

    # Original initialization:
    # [40, 1e5, 1e5] repeated for 3 horizons.
    running_minimum = np.asarray([40.0, 1e5, 1e5] * 3, dtype=np.float64)

    trigger_records = []
    best_records = []

    for _, row in history.iterrows():
        epoch = int(row["epoch"])

        validation_vector = build_validation_vector(row)

        improved_mask = validation_vector < running_minimum

        improved_indices = np.flatnonzero(improved_mask)

        improved_names = [METRICS[index][0] for index in improved_indices]

        if improved_indices.size > 0:
            previous_values = running_minimum[improved_indices].copy()

            running_minimum[improved_indices] = validation_vector[improved_indices]

            trigger_records.append(
                {
                    "epoch": epoch,
                    "improved_metric_count": int(improved_indices.size),
                    "improved_metrics": "; ".join(improved_names),
                    "checkpoint_exists": checkpoint_path_for_epoch(
                        checkpoint_directory, epoch
                    ).exists(),
                }
            )

            for local_index, metric_index in enumerate(improved_indices):
                metric_name, _, _ = METRICS[metric_index]

                best_records.append(
                    {
                        "epoch": epoch,
                        "metric": metric_name,
                        "previous_best": float(previous_values[local_index]),
                        "new_best": float(validation_vector[metric_index]),
                    }
                )

    triggers = pd.DataFrame(trigger_records)

    improvements = pd.DataFrame(best_records)

    if triggers.empty:
        raise RuntimeError("No validation improvements were detected.")

    last_trigger_epoch = int(triggers.iloc[-1]["epoch"])

    final_epoch = int(history["epoch"].max())

    last_trigger_checkpoint = checkpoint_path_for_epoch(
        checkpoint_directory, last_trigger_epoch
    )

    final_checkpoint = checkpoint_path_for_epoch(checkpoint_directory, final_epoch)

    # Best epoch for each individual metric.
    best_metric_rows = []

    for metric_name, column_name, _ in METRICS:
        best_index = history[column_name].idxmin()

        best_row = history.loc[best_index]

        best_metric_rows.append(
            {
                "metric": metric_name,
                "best_epoch": int(best_row["epoch"]),
                "best_value_in_history": float(best_row[column_name]),
            }
        )

    best_metrics = pd.DataFrame(best_metric_rows)

    output_directory = result_directory / "selection_protocol"

    output_directory.mkdir(parents=True, exist_ok=True)

    triggers_path = output_directory / "original_protocol_trigger_epochs.csv"

    improvements_path = output_directory / "original_protocol_metric_improvements.csv"

    best_metrics_path = output_directory / "best_validation_epochs.csv"

    summary_path = output_directory / "selection_protocol_summary.txt"

    triggers.to_csv(triggers_path, index=False)

    improvements.to_csv(improvements_path, index=False)

    best_metrics.to_csv(best_metrics_path, index=False)

    summary_lines = [
        "Original STGCN Validation-Selection Protocol Audit",
        "=" * 62,
        "",
        f"Run name: {args.run_name}",
        f"Final training epoch: {final_epoch}",
        f"Number of trigger epochs: {len(triggers)}",
        f"Last trigger epoch: {last_trigger_epoch}",
        ("Last-trigger checkpoint exists: " f"{last_trigger_checkpoint.exists()}"),
        ("Final checkpoint exists: " f"{final_checkpoint.exists()}"),
        "",
        "Original repository behavior:",
        ("1. Validation metrics are tracked independently."),
        (
            "2. If any validation metric improves, the complete "
            "test metric vector is recomputed and replaces the "
            "previous test vector."
        ),
        (
            "3. Therefore, the final validation-minimum vector can "
            "combine values from different epochs."
        ),
        (
            "4. The final reported test vector corresponds to the "
            "last epoch that improved at least one validation metric, "
            "not necessarily one checkpoint minimizing all metrics."
        ),
        "",
    ]

    if last_trigger_epoch == final_epoch:
        summary_lines.extend(
            [
                "Conclusion:",
                (
                    "The last original-protocol trigger occurs at "
                    "the final epoch. Therefore the epoch-50 test "
                    "result is consistent with the original tracking "
                    "logic for this run."
                ),
            ]
        )
    elif last_trigger_checkpoint.exists():
        summary_lines.extend(
            [
                "Conclusion:",
                (
                    "The last trigger occurs before the final epoch, "
                    "and its checkpoint exists. Evaluate that checkpoint "
                    "to reproduce the original tracking logic."
                ),
            ]
        )
    else:
        summary_lines.extend(
            [
                "Conclusion:",
                (
                    "The last trigger occurs before the final epoch, "
                    "but its checkpoint was not saved. A rerun with "
                    "per-epoch test tracking or per-epoch checkpoints "
                    "is required for exact reproduction."
                ),
            ]
        )

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 78)
    print("Original STGCN selection-protocol audit")
    print("=" * 78)

    print(triggers.to_string(index=False))

    print("\n" + "=" * 78)
    print("Best validation epoch per metric")
    print("=" * 78)

    print(best_metrics.to_string(index=False))

    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)

    print(f"Final epoch: {final_epoch}")
    print(f"Last trigger epoch: " f"{last_trigger_epoch}")
    print("Last-trigger checkpoint exists:", last_trigger_checkpoint.exists())

    if last_trigger_epoch == final_epoch:
        print(
            "The epoch-50 test result is consistent "
            "with the original tracking logic for "
            "this run."
        )
    elif last_trigger_checkpoint.exists():
        print(
            "Evaluate the last-trigger checkpoint " "before comparing with the paper."
        )
    else:
        print(
            "An exact protocol rerun is required "
            "because the last-trigger checkpoint "
            "does not exist."
        )

    print("\nGenerated files:")
    print(triggers_path)
    print(improvements_path)
    print(best_metrics_path)
    print(summary_path)


if __name__ == "__main__":
    main()
