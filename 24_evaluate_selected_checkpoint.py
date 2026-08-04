from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Windows 环境下必须优先导入 PyTorch
import torch

import pandas as pd

from model.paper_model import PaperSTGCN
from paper_main import (
    evaluate_autoregressive,
    prepare_graph,
    print_metrics,
)
from script import dataloader


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a selected STGCN checkpoint."
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def save_metrics(metrics, output_path: Path):
    rows = []

    for horizon in (3, 6, 9):
        rows.append(
            {
                "horizon_steps": horizon,
                "minutes": horizon * 5,
                "MAPE_percent":
                    metrics[horizon]["MAPE"] * 100,
                "MAE": metrics[horizon]["MAE"],
                "RMSE": metrics[horizon]["RMSE"],
            }
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    args = get_arguments()

    project_root = Path(__file__).resolve().parent

    checkpoint_path = (
        project_root
        / "checkpoints"
        / args.run_name
        / f"paper_stgcn_epoch_{args.epoch:03d}.pt"
    )

    result_directory = (
        project_root
        / "results"
        / args.run_name
    )

    selection_directory = (
        result_directory
        / "selection_protocol"
    )

    selection_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    saved_args = checkpoint["arguments"]

    dataset = saved_args["dataset"]
    graph_source = saved_args["graph_source"]
    gso_source = saved_args["gso_source"]
    n_his = int(saved_args["n_his"])
    n_pred = int(saved_args["n_pred"])
    Kt = int(saved_args["Kt"])
    Ks = int(saved_args["Ks"])
    batch_size = int(saved_args["batch_size"])

    print("=" * 78)
    print("Selected STGCN checkpoint evaluation")
    print("=" * 78)
    print("Device:", device)
    print("Run name:", args.run_name)
    print("Requested epoch:", args.epoch)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint path:", checkpoint_path)
    print("Graph source:", graph_source)
    print("GSO source:", gso_source)

    (
        gso,
        n_vertex,
        graph_info,
    ) = prepare_graph(
        dataset=dataset,
        graph_source=graph_source,
        device=device,
        gso_source=gso_source,
    )

    print("\nGraph:")
    print(
        "Adjacency non-zero values:",
        graph_info["nonzero_count"],
    )
    print(
        "Isolated-node indices:",
        graph_info["isolated_node_indices"],
    )
    print(
        "Largest eigenvalue:",
        graph_info["lambda_max"],
    )

    (
        scaler,
        _x_train,
        _y_train,
        _val_sequences,
        test_sequences,
    ) = dataloader.prepare_autoregressive_data(
        dataset_name=dataset,
        n_his=n_his,
        max_pred_steps=n_pred,
        device=device,
        n_train_days=34,
        n_val_days=5,
        n_test_days=5,
        day_slot=288,
    )

    if abs(
        float(scaler.mean)
        - float(checkpoint["scaler_mean"])
    ) > 1e-8:
        raise ValueError(
            "Scaler mean does not match the checkpoint."
        )

    if abs(
        float(scaler.std)
        - float(checkpoint["scaler_std"])
    ) > 1e-8:
        raise ValueError(
            "Scaler std does not match the checkpoint."
        )

    model = PaperSTGCN(
        Kt=Kt,
        Ks=Ks,
        n_his=n_his,
        n_vertex=n_vertex,
        gso=gso,
        droprate=0.0,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    metrics = evaluate_autoregressive(
        model=model,
        sequences=test_sequences,
        scaler=scaler,
        n_his=n_his,
        n_pred=n_pred,
        batch_size=batch_size,
    )

    print("\n" + "=" * 78)
    print(
        f"Epoch {args.epoch} autoregressive test"
    )
    print("=" * 78)

    print_metrics(
        "Test:",
        metrics,
    )

    metrics_path = (
        selection_directory
        / (
            f"selected_checkpoint_epoch_"
            f"{args.epoch:03d}_metrics.csv"
        )
    )

    selected_rows = save_metrics(
        metrics,
        metrics_path,
    )

    final_metrics_path = (
        result_directory
        / "final_metrics.csv"
    )

    comparison_path = None

    if final_metrics_path.exists():
        final_frame = pd.read_csv(
            final_metrics_path
        )

        selected_by_minutes = {
            int(row["minutes"]): row
            for row in selected_rows
        }

        comparison_rows = []

        for _, final_row in final_frame.iterrows():
            minutes = int(final_row["minutes"])
            selected = selected_by_minutes[minutes]

            comparison_rows.append(
                {
                    "minutes": minutes,
                    "selected_epoch": args.epoch,
                    "selected_MAPE_percent":
                        selected["MAPE_percent"],
                    "final_epoch_MAPE_percent":
                        float(final_row["MAPE_percent"]),
                    "MAPE_difference_pp":
                        selected["MAPE_percent"]
                        - float(final_row["MAPE_percent"]),
                    "selected_MAE":
                        selected["MAE"],
                    "final_epoch_MAE":
                        float(final_row["MAE"]),
                    "MAE_difference":
                        selected["MAE"]
                        - float(final_row["MAE"]),
                    "selected_RMSE":
                        selected["RMSE"],
                    "final_epoch_RMSE":
                        float(final_row["RMSE"]),
                    "RMSE_difference":
                        selected["RMSE"]
                        - float(final_row["RMSE"]),
                }
            )

        comparison_path = (
            selection_directory
            / (
                f"selected_checkpoint_epoch_"
                f"{args.epoch:03d}_vs_final.csv"
            )
        )

        with comparison_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    comparison_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                comparison_rows
            )

        print("\n" + "=" * 78)
        print("Selected checkpoint vs final epoch")
        print("=" * 78)

        for row in comparison_rows:
            print(
                f"{row['minutes']:02d} min | "
                f"MAPE {row['MAPE_difference_pp']:+.6f} pp | "
                f"MAE {row['MAE_difference']:+.6f} | "
                f"RMSE {row['RMSE_difference']:+.6f}"
            )

    summary = {
        "run_name": args.run_name,
        "selected_epoch": args.epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset": dataset,
        "graph_source": graph_source,
        "gso_source": gso_source,
        "test_metrics": {
            str(row["minutes"]): {
                "MAPE_percent":
                    row["MAPE_percent"],
                "MAE":
                    row["MAE"],
                "RMSE":
                    row["RMSE"],
            }
            for row in selected_rows
        },
    }

    summary_path = (
        selection_directory
        / (
            f"selected_checkpoint_epoch_"
            f"{args.epoch:03d}_summary.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved files:")
    print(metrics_path)

    if comparison_path is not None:
        print(comparison_path)

    print(summary_path)


if __name__ == "__main__":
    main()
