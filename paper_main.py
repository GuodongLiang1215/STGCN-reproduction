import argparse
import csv
import json
import random
import time
import warnings
from pathlib import Path

# Import PyTorch before NumPy on Windows to avoid DLL load-order issues.
import torch

import numpy as np
import scipy.sparse as sp

from model.paper_model import PaperSTGCN
from script import dataloader, utility


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch and request deterministic cuDNN kernels."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser(
        description="STGCN reproduction with an auditable graph pipeline"
    )

    parser.add_argument("--dataset", type=str, default="pemsd7-m")

    parser.add_argument(
        "--graph_source",
        type=str,
        choices=("current", "official"),
        default="official",
        help=(
            "Choose the adjacency matrix: "
            "'current' uses adj.npz; "
            "'official' uses adj_official_stgcn.npz."
        ),
    )

    parser.add_argument(
        "--gso_source",
        type=str,
        choices=("current", "original"),
        default="original",
        help=(
            "Choose the GSO construction: "
            "'current' uses the modern PyTorch "
            "utility; 'original' uses the "
            "TensorFlow STGCN scaled Laplacian."
        ),
    )

    parser.add_argument("--n_his", type=int, default=12)

    parser.add_argument(
        "--n_pred", type=int, default=9, help="Maximum autoregressive prediction steps"
    )

    parser.add_argument("--Kt", type=int, default=3)

    parser.add_argument("--Ks", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--lr_step_size", type=int, default=5)

    parser.add_argument("--lr_gamma", type=float, default=0.7)

    parser.add_argument("--checkpoint_every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--run_name", type=str, default="official_original_gso_seed42"
    )

    return parser.parse_args()


def prepare_graph(dataset, graph_source, device, gso_source="current"):
    """Load the requested adjacency matrix and construct its scaled GSO.

    `current` selects the third-party PyTorch graph or utility. `official` and
    `original` select the graph and scaled-Laplacian formulation distributed
    with the TensorFlow reference implementation.
    """

    dataset_directory = Path("data") / dataset

    if graph_source == "current":
        graph_path = dataset_directory / "adj.npz"

        adj, n_vertex = dataloader.load_adj(dataset)

    elif graph_source == "official":
        if dataset != "pemsd7-m":
            raise ValueError(
                "The official graph option is "
                "currently available only for "
                "pemsd7-m."
            )

        graph_path = dataset_directory / "adj_official_stgcn.npz"

        if not graph_path.exists():
            raise FileNotFoundError(
                "Official STGCN adjacency matrix "
                "was not found:\n"
                f"{graph_path}\n\n"
                "Run `python -m "
                "experiments.verification.prepare_official_adjacency` first."
            )

        adj = sp.load_npz(graph_path).tocsc()

        n_vertex = adj.shape[0]

    else:
        raise ValueError(f"Unsupported graph source: " f"{graph_source}")

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            "The adjacency matrix must be square. " f"Current shape: {adj.shape}"
        )

    if adj.shape[0] != n_vertex:
        raise ValueError(
            "The number of graph nodes does not " "match the adjacency matrix shape."
        )

    degree = np.asarray(adj.sum(axis=1)).reshape(-1)

    isolated_node_indices = np.flatnonzero(degree == 0).tolist()

    if gso_source == "current":
        gso_sparse = utility.calc_gso(adj, "sym_norm_lap")

        gso_sparse = utility.calc_chebynet_gso(gso_sparse)

        lambda_max = None

    elif gso_source == "original":
        if graph_source != "official":
            raise ValueError(
                "gso_source='original' must be " "used with graph_source='official'."
            )

        if not hasattr(utility, "calc_original_stgcn_gso"):
            raise AttributeError(
                "script.utility does not contain "
                "calc_original_stgcn_gso(). "
                "Please add the original STGCN "
                "scaled-Laplacian implementation "
                "to script/utility.py first."
            )

        (gso_sparse, lambda_max) = utility.calc_original_stgcn_gso(adj)

    else:
        raise ValueError(f"Unsupported GSO source: " f"{gso_source}")

    if sp.issparse(gso_sparse):
        gso_array = gso_sparse.toarray().astype(np.float32)
    else:
        gso_array = np.asarray(gso_sparse, dtype=np.float32)

    if not np.isfinite(gso_array).all():
        raise ValueError("The generated GSO contains NaN " "or infinite values.")

    gso = torch.from_numpy(gso_array).to(device)

    graph_info = {
        "source": graph_source,
        "path": str(graph_path),
        "shape": tuple(adj.shape),
        "nonzero_count": int(adj.nnz),
        "diagonal_nonzero_count": int(np.count_nonzero(adj.diagonal())),
        "gso_source": gso_source,
        "isolated_node_count": len(isolated_node_indices),
        "isolated_node_indices": isolated_node_indices,
        "lambda_max": lambda_max,
        "gso_minimum": float(gso_array.min()),
        "gso_maximum": float(gso_array.max()),
        "gso_finite": bool(np.isfinite(gso_array).all()),
    }

    return (gso, n_vertex, graph_info)


def paper_l2_loss(prediction, target):
    """TensorFlow `tf.nn.l2_loss`: summed half-squared error, not mean MSE."""

    return 0.5 * torch.sum((prediction - target) ** 2)


def shuffled_batches(x, y, batch_size):
    """Shuffle each epoch and retain the final partial batch (`dynamic_batch`)."""

    sample_count = len(x)

    indices = torch.randperm(sample_count, device=x.device)

    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)

        batch_indices = indices[start:end]

        yield (x[batch_indices], y[batch_indices])


@torch.no_grad()
def evaluate_autoregressive(
    model, sequences, scaler, n_his, n_pred, batch_size, horizons=(3, 6, 9)
):
    """Run recursive inference and accumulate metrics at selected horizons."""

    model.eval()

    metric_sums = {
        horizon: {
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "percentage_error": 0.0,
            "count": 0,
        }
        for horizon in horizons
    }

    sample_count = len(sequences)

    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)

        sequence_batch = sequences[start:end]

        # Start with the normalized history in [batch, time, node] order.
        history = sequence_batch[:, 0:n_his, :, 0]

        for step in range(1, n_pred + 1):
            model_input = history.unsqueeze(1)

            prediction = model(model_input).view(len(sequence_batch), -1)

            if step in horizons:
                target_index = n_his + step - 1

                target = sequence_batch[:, target_index, :, 0]

                prediction_original = prediction * scaler.std + scaler.mean

                target_original = target * scaler.std + scaler.mean

                difference = torch.abs(prediction_original - target_original)

                metric_sums[step]["absolute_error"] += difference.sum().item()

                metric_sums[step]["squared_error"] += (difference**2).sum().item()

                # Match the reference MAPE denominator convention.
                metric_sums[step]["percentage_error"] += (
                    (difference / (target_original + 1e-5)).sum().item()
                )

                metric_sums[step]["count"] += difference.numel()

            # Feed the prediction back into the next history window.
            history = torch.cat([history[:, 1:, :], prediction.unsqueeze(1)], dim=1)

    metrics = {}

    for horizon in horizons:
        count = metric_sums[horizon]["count"]

        mae = metric_sums[horizon]["absolute_error"] / count

        rmse = np.sqrt(metric_sums[horizon]["squared_error"] / count)

        mape = metric_sums[horizon]["percentage_error"] / count

        metrics[horizon] = {"MAPE": float(mape), "MAE": float(mae), "RMSE": float(rmse)}

    return metrics


def print_metrics(title, metrics):
    print(title)

    for horizon in (3, 6, 9):
        minutes = horizon * 5
        result = metrics[horizon]

        print(
            f"{minutes:02d} min | "
            f"MAPE {result['MAPE'] * 100:7.3f}% | "
            f"MAE {result['MAE']:7.3f} | "
            f"RMSE {result['RMSE']:7.3f}"
        )


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, args):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "arguments": vars(args),
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")


def save_history(history, output_path):
    if not history:
        return

    fieldnames = list(history[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(history)


def save_final_metrics(metrics, output_path):
    rows = []

    for horizon in (3, 6, 9):
        rows.append(
            {
                "horizon_steps": horizon,
                "minutes": horizon * 5,
                "MAPE_percent": metrics[horizon]["MAPE"] * 100,
                "MAE": metrics[horizon]["MAE"],
                "RMSE": metrics[horizon]["RMSE"],
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))

        writer.writeheader()
        writer.writerows(rows)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    warnings.filterwarnings("ignore", category=FutureWarning)

    args = get_arguments()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_directory = Path("checkpoints") / args.run_name

    result_directory = Path("results") / args.run_name

    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    result_directory.mkdir(parents=True, exist_ok=True)

    config_path = result_directory / "config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print("=" * 75)
    print("Paper-aligned STGCN")
    print("=" * 75)

    print("Device:", device)
    print("Run name:", args.run_name)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Optimizer: RMSProp")
    print("Learning rate:", args.lr)

    (gso, n_vertex, graph_info) = prepare_graph(
        dataset=args.dataset,
        graph_source=args.graph_source,
        device=device,
        gso_source=args.gso_source,
    )

    print("\nGraph:")
    print("Graph source:", graph_info["source"])
    print("Graph path:", graph_info["path"])
    print("Graph shape:", graph_info["shape"])
    print("Adjacency non-zero values:", graph_info["nonzero_count"])
    print("Adjacency non-zero diagonal values:", graph_info["diagonal_nonzero_count"])
    print("GSO source:", graph_info["gso_source"])

    print("Isolated-node count:", graph_info["isolated_node_count"])

    print("Isolated-node indices:", graph_info["isolated_node_indices"])

    if graph_info["lambda_max"] is not None:
        print("Largest Laplacian eigenvalue:", graph_info["lambda_max"])

    (scaler, x_train, y_train, val_sequences, test_sequences) = (
        dataloader.prepare_autoregressive_data(
            dataset_name=args.dataset,
            n_his=args.n_his,
            max_pred_steps=args.n_pred,
            device=device,
            n_train_days=34,
            n_val_days=5,
            n_test_days=5,
            day_slot=288,
        )
    )

    print("\nData:")
    print("Train X:", x_train.shape)
    print("Train y:", y_train.shape)
    print("Validation:", val_sequences.shape)
    print("Test:", test_sequences.shape)

    print(f"Global mean: {scaler.mean:.6f}")

    print(f"Global std:  {scaler.std:.6f}")

    model = PaperSTGCN(
        Kt=args.Kt,
        Ks=args.Ks,
        n_his=args.n_his,
        n_vertex=n_vertex,
        gso=gso,
        droprate=0.0,
    ).to(device)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        alpha=0.9,
        eps=1e-10,
        momentum=0.0,
        weight_decay=0.0,
        centered=False,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("Model parameters:", parameter_count)

    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()

        epoch_loss_sum = 0.0
        epoch_copy_loss_sum = 0.0
        processed_samples = 0

        learning_rate_used = optimizer.param_groups[0]["lr"]

        for batch_number, (x_batch, y_batch) in enumerate(
            shuffled_batches(x_train, y_train, args.batch_size), start=1
        ):
            optimizer.zero_grad()

            prediction = model(x_batch).view(len(x_batch), n_vertex)

            loss = paper_l2_loss(prediction, y_batch)

            # Reference persistence baseline: copy the last observed speed.
            copy_loss = paper_l2_loss(x_batch[:, 0, -1, :], y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_copy_loss_sum += copy_loss.item()

            processed_samples += len(x_batch)

            if batch_number == 1 or batch_number % 50 == 0:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Batch {batch_number:03d} | "
                    f"L2 {loss.item():.3f} | "
                    f"Copy {copy_loss.item():.3f}"
                )

        validation_metrics = evaluate_autoregressive(
            model=model,
            sequences=val_sequences,
            scaler=scaler,
            n_his=args.n_his,
            n_pred=args.n_pred,
            batch_size=args.batch_size,
        )

        epoch_seconds = time.time() - epoch_start

        average_l2_per_sample = epoch_loss_sum / processed_samples

        average_copy_per_sample = epoch_copy_loss_sum / processed_samples

        print("\n" + "-" * 75)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={learning_rate_used:.10f} | "
            f"L2/sample={average_l2_per_sample:.6f} | "
            f"Copy/sample={average_copy_per_sample:.6f} | "
            f"Time={epoch_seconds:.2f}s"
        )

        print_metrics("Validation:", validation_metrics)

        print("-" * 75 + "\n")

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate_used,
                "train_l2_per_sample": average_l2_per_sample,
                "copy_l2_per_sample": average_copy_per_sample,
                "val_15_mape_percent": validation_metrics[3]["MAPE"] * 100,
                "val_15_mae": validation_metrics[3]["MAE"],
                "val_15_rmse": validation_metrics[3]["RMSE"],
                "val_30_mape_percent": validation_metrics[6]["MAPE"] * 100,
                "val_30_mae": validation_metrics[6]["MAE"],
                "val_30_rmse": validation_metrics[6]["RMSE"],
                "val_45_mape_percent": validation_metrics[9]["MAPE"] * 100,
                "val_45_mae": validation_metrics[9]["MAE"],
                "val_45_rmse": validation_metrics[9]["RMSE"],
                "epoch_seconds": epoch_seconds,
            }
        )

        save_history(history, result_directory / "training_history.csv")

        if epoch % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_directory / (
                f"paper_stgcn_" f"epoch_{epoch:03d}.pt"
            )

            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
            )

        # Step after the epoch to preserve the learning rate used in its log row.
        scheduler.step()

    final_checkpoint_path = checkpoint_directory / (
        f"paper_stgcn_" f"epoch_{args.epochs:03d}.pt"
    )

    if not final_checkpoint_path.exists():
        save_checkpoint(
            path=final_checkpoint_path,
            epoch=args.epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
        )

    print("\n" + "=" * 75)
    print("Final autoregressive test")
    print("=" * 75)

    final_test_metrics = evaluate_autoregressive(
        model=model,
        sequences=test_sequences,
        scaler=scaler,
        n_his=args.n_his,
        n_pred=args.n_pred,
        batch_size=args.batch_size,
    )

    print_metrics("Test:", final_test_metrics)

    save_final_metrics(final_test_metrics, result_directory / "final_metrics.csv")

    print("\nSaved files:")
    print(final_checkpoint_path)

    print(result_directory / "training_history.csv")

    print(result_directory / "final_metrics.csv")

    print(result_directory / "config.json")


if __name__ == "__main__":
    main()
