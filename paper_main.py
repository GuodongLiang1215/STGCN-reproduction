import argparse
import csv
import json
import random
import time
import warnings
from pathlib import Path

# Windows 下优先导入 PyTorch
import torch

import numpy as np
import scipy.sparse as sp

from model.paper_model import PaperSTGCN
from script import dataloader, utility

# =========================================================
# 1. 随机种子
# =========================================================


def set_random_seed(seed):
    """
    尽量让每次实验结果更加稳定。
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================================================
# 2. 命令行参数
# =========================================================


def get_arguments():
    parser = argparse.ArgumentParser(description="Paper-aligned STGCN reproduction")

    parser.add_argument(
        "--dataset",
        type=str,
        default="pemsd7-m",
    )

    parser.add_argument(
        "--graph_source",
        type=str,
        choices=(
            "current",
            "official",
        ),
        default="current",
        help=(
            "Choose the adjacency matrix: "
            "'current' uses adj.npz; "
            "'official' uses adj_official_stgcn.npz."
        ),
    )

    parser.add_argument(
        "--gso_source",
        type=str,
        choices=(
            "current",
            "original",
        ),
        default="current",
        help=(
            "Choose the GSO construction: "
            "'current' uses the modern PyTorch "
            "utility; 'original' uses the "
            "TensorFlow STGCN scaled Laplacian."
        ),
    )

    parser.add_argument(
        "--n_his",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--n_pred",
        type=int,
        default=9,
        help="Maximum autoregressive prediction steps",
    )

    parser.add_argument(
        "--Kt",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--Ks",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="paper_50epochs",
    )

    return parser.parse_args()


# =========================================================
# 3. 构建图算子
# =========================================================


def prepare_graph(
    dataset,
    graph_source,
    device,
    gso_source="current",
):
    """
    根据 graph_source 选择邻接矩阵，并根据 gso_source
    构造图卷积使用的缩放图算子（GSO）。

    graph_source="current":
        data/<dataset>/adj.npz

    graph_source="official":
        data/<dataset>/adj_official_stgcn.npz

    gso_source="current":
        使用当前 PyTorch 工具中的归一化 Laplacian
        与 Chebyshev 缩放方法。

    gso_source="original":
        使用原始 TensorFlow STGCN 的
        scaled_laplacian() 构造方法。
    """

    dataset_directory = Path("data") / dataset

    # -----------------------------------------------------
    # 1. 选择邻接矩阵
    # -----------------------------------------------------

    if graph_source == "current":
        graph_path = dataset_directory / "adj.npz"

        adj, n_vertex = dataloader.load_adj(
            dataset
        )

    elif graph_source == "official":
        if dataset != "pemsd7-m":
            raise ValueError(
                "The official graph option is "
                "currently available only for "
                "pemsd7-m."
            )

        graph_path = (
            dataset_directory
            / "adj_official_stgcn.npz"
        )

        if not graph_path.exists():
            raise FileNotFoundError(
                "Official STGCN adjacency matrix "
                "was not found:\n"
                f"{graph_path}\n\n"
                "Run "
                "18_prepare_official_adjacency.py "
                "first."
            )

        adj = sp.load_npz(
            graph_path
        ).tocsc()

        n_vertex = adj.shape[0]

    else:
        raise ValueError(
            f"Unsupported graph source: "
            f"{graph_source}"
        )

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            "The adjacency matrix must be square. "
            f"Current shape: {adj.shape}"
        )

    if adj.shape[0] != n_vertex:
        raise ValueError(
            "The number of graph nodes does not "
            "match the adjacency matrix shape."
        )

    # -----------------------------------------------------
    # 2. 检查孤立节点
    # -----------------------------------------------------

    degree = np.asarray(
        adj.sum(axis=1)
    ).reshape(-1)

    isolated_node_indices = (
        np.flatnonzero(
            degree == 0
        ).tolist()
    )

    # -----------------------------------------------------
    # 3. 构造 GSO
    # -----------------------------------------------------

    if gso_source == "current":
        gso_sparse = utility.calc_gso(
            adj,
            "sym_norm_lap",
        )

        gso_sparse = (
            utility.calc_chebynet_gso(
                gso_sparse
            )
        )

        lambda_max = None

    elif gso_source == "original":
        if graph_source != "official":
            raise ValueError(
                "gso_source='original' must be "
                "used with graph_source='official'."
            )

        if not hasattr(
            utility,
            "calc_original_stgcn_gso",
        ):
            raise AttributeError(
                "script.utility does not contain "
                "calc_original_stgcn_gso(). "
                "Please add the original STGCN "
                "scaled-Laplacian implementation "
                "to script/utility.py first."
            )

        (
            gso_sparse,
            lambda_max,
        ) = utility.calc_original_stgcn_gso(
            adj
        )

    else:
        raise ValueError(
            f"Unsupported GSO source: "
            f"{gso_source}"
        )

    # -----------------------------------------------------
    # 4. 转换为 PyTorch 张量
    # -----------------------------------------------------

    if sp.issparse(gso_sparse):
        gso_array = (
            gso_sparse
            .toarray()
            .astype(np.float32)
        )
    else:
        gso_array = np.asarray(
            gso_sparse,
            dtype=np.float32,
        )

    if not np.isfinite(
        gso_array
    ).all():
        raise ValueError(
            "The generated GSO contains NaN "
            "or infinite values."
        )

    gso = torch.from_numpy(
        gso_array
    ).to(device)

    # -----------------------------------------------------
    # 5. 保存图信息
    # -----------------------------------------------------

    graph_info = {
        "source": graph_source,
        "path": str(graph_path),
        "shape": tuple(adj.shape),
        "nonzero_count": int(
            adj.nnz
        ),
        "diagonal_nonzero_count": int(
            np.count_nonzero(
                adj.diagonal()
            )
        ),
        "gso_source": gso_source,
        "isolated_node_count": len(
            isolated_node_indices
        ),
        "isolated_node_indices":
            isolated_node_indices,
        "lambda_max": lambda_max,
        "gso_minimum": float(
            gso_array.min()
        ),
        "gso_maximum": float(
            gso_array.max()
        ),
        "gso_finite": bool(
            np.isfinite(
                gso_array
            ).all()
        ),
    }

    return (
        gso,
        n_vertex,
        graph_info,
    )


# =========================================================
# 4. 论文式 L2 Loss
# =========================================================


def paper_l2_loss(prediction, target):
    """
    等价于 TensorFlow 的 tf.nn.l2_loss：

        0.5 × sum((prediction - target)^2)

    注意：这里不是平均 MSE。
    """

    return 0.5 * torch.sum((prediction - target) ** 2)


# =========================================================
# 5. 生成随机训练 batch
# =========================================================


def shuffled_batches(
    x,
    y,
    batch_size,
):
    """
    每一个 epoch 都重新打乱训练样本。

    最后不足 batch_size 的样本也会保留，
    对应原作者 dynamic_batch=True。
    """

    sample_count = len(x)

    indices = torch.randperm(
        sample_count,
        device=x.device,
    )

    for start in range(
        0,
        sample_count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            sample_count,
        )

        batch_indices = indices[start:end]

        yield (
            x[batch_indices],
            y[batch_indices],
        )


# =========================================================
# 6. 递归多步评估
# =========================================================


@torch.no_grad()
def evaluate_autoregressive(
    model,
    sequences,
    scaler,
    n_his,
    n_pred,
    batch_size,
    horizons=(3, 6, 9),
):
    """
    把模型递归运行 n_pred 次。

    第1次预测：
        5分钟后

    第3次预测：
        15分钟后

    第6次预测：
        30分钟后

    第9次预测：
        45分钟后
    """

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

    for start in range(
        0,
        sample_count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            sample_count,
        )

        sequence_batch = sequences[start:end]

        # 初始历史窗口：
        # [batch, 12, node]
        history = sequence_batch[
            :,
            0:n_his,
            :,
            0,
        ]

        for step in range(1, n_pred + 1):
            # 模型需要：
            # [batch, channel, time, node]
            model_input = history.unsqueeze(1)

            prediction = model(model_input).view(
                len(sequence_batch),
                -1,
            )

            if step in horizons:
                target_index = n_his + step - 1

                target = sequence_batch[
                    :,
                    target_index,
                    :,
                    0,
                ]

                # 从标准化空间恢复到真实交通速度
                prediction_original = prediction * scaler.std + scaler.mean

                target_original = target * scaler.std + scaler.mean

                difference = torch.abs(prediction_original - target_original)

                metric_sums[step]["absolute_error"] += difference.sum().item()

                metric_sums[step]["squared_error"] += (difference**2).sum().item()

                # 与原作者公式保持一致
                metric_sums[step]["percentage_error"] += (
                    (difference / (target_original + 1e-5)).sum().item()
                )

                metric_sums[step]["count"] += difference.numel()

            # 把预测结果放入历史窗口末尾
            history = torch.cat(
                [
                    history[:, 1:, :],
                    prediction.unsqueeze(1),
                ],
                dim=1,
            )

    metrics = {}

    for horizon in horizons:
        count = metric_sums[horizon]["count"]

        mae = metric_sums[horizon]["absolute_error"] / count

        rmse = np.sqrt(metric_sums[horizon]["squared_error"] / count)

        mape = metric_sums[horizon]["percentage_error"] / count

        metrics[horizon] = {
            "MAPE": float(mape),
            "MAE": float(mae),
            "RMSE": float(rmse),
        }

    return metrics


# =========================================================
# 7. 打印递归指标
# =========================================================


def print_metrics(
    title,
    metrics,
):
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


# =========================================================
# 8. 保存 checkpoint
# =========================================================


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "arguments": vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )

    print(f"Checkpoint saved: {path}")


# =========================================================
# 9. 保存训练历史
# =========================================================


def save_history(
    history,
    output_path,
):
    if not history:
        return

    fieldnames = list(history[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


# =========================================================
# 10. 保存最终指标
# =========================================================


def save_final_metrics(
    metrics,
    output_path,
):
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


# =========================================================
# 11. 主程序
# =========================================================


def main():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    args = get_arguments()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_directory = Path("checkpoints") / args.run_name

    result_directory = Path("results") / args.run_name

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = result_directory / "config.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
        )

    print("=" * 75)
    print("Paper-aligned STGCN")
    print("=" * 75)

    print("Device:", device)
    print("Run name:", args.run_name)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Optimizer: RMSProp")
    print("Learning rate:", args.lr)

    # -----------------------------------------------------
    # 图结构
    # -----------------------------------------------------

    (
        gso,
        n_vertex,
        graph_info,
    ) = prepare_graph(
        dataset=args.dataset,
        graph_source=args.graph_source,
        device=device,
        gso_source=args.gso_source,
    )

    print("\nGraph:")
    print(
        "Graph source:",
        graph_info["source"],
    )
    print(
        "Graph path:",
        graph_info["path"],
    )
    print(
        "Graph shape:",
        graph_info["shape"],
    )
    print(
        "Adjacency non-zero values:",
        graph_info["nonzero_count"],
    )
    print(
        "Adjacency non-zero diagonal values:",
        graph_info["diagonal_nonzero_count"],
    )
    print(
        "GSO source:",
        graph_info["gso_source"],
    )

    print(
        "Isolated-node count:",
        graph_info[
            "isolated_node_count"
        ],
    )

    print(
        "Isolated-node indices:",
        graph_info[
            "isolated_node_indices"
        ],
    )

    if graph_info["lambda_max"] is not None:
        print(
            "Largest Laplacian eigenvalue:",
            graph_info["lambda_max"],
        )

    # -----------------------------------------------------
    # 数据
    # -----------------------------------------------------

    (
        scaler,
        x_train,
        y_train,
        val_sequences,
        test_sequences,
    ) = dataloader.prepare_autoregressive_data(
        dataset_name=args.dataset,
        n_his=args.n_his,
        max_pred_steps=args.n_pred,
        device=device,
        n_train_days=34,
        n_val_days=5,
        n_test_days=5,
        day_slot=288,
    )

    print("\nData:")
    print("Train X:", x_train.shape)
    print("Train y:", y_train.shape)
    print("Validation:", val_sequences.shape)
    print("Test:", test_sequences.shape)

    print(f"Global mean: {scaler.mean:.6f}")

    print(f"Global std:  {scaler.std:.6f}")

    # -----------------------------------------------------
    # 模型
    # -----------------------------------------------------

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
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        "Model parameters:",
        parameter_count,
    )

    history = []

    # -----------------------------------------------------
    # 训练
    # -----------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        model.train()

        epoch_loss_sum = 0.0
        epoch_copy_loss_sum = 0.0
        processed_samples = 0

        learning_rate_used = optimizer.param_groups[0]["lr"]

        for batch_number, (
            x_batch,
            y_batch,
        ) in enumerate(
            shuffled_batches(
                x_train,
                y_train,
                args.batch_size,
            ),
            start=1,
        ):
            optimizer.zero_grad()

            prediction = model(x_batch).view(
                len(x_batch),
                n_vertex,
            )

            loss = paper_l2_loss(
                prediction,
                y_batch,
            )

            # 原作者打印的 copy loss：
            # 使用最后一个历史速度直接预测下一时刻
            copy_loss = paper_l2_loss(
                x_batch[:, 0, -1, :],
                y_batch,
            )

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

        # -------------------------------------------------
        # 每轮执行递归验证
        # -------------------------------------------------

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

        print_metrics(
            "Validation:",
            validation_metrics,
        )

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

        save_history(
            history,
            result_directory / "training_history.csv",
        )

        # 原作者每10轮保存一次
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

        # 当前 epoch 完成后再更新学习率
        scheduler.step()

    # -----------------------------------------------------
    # 确保最后一轮模型被保存
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 最终测试
    # -----------------------------------------------------

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

    print_metrics(
        "Test:",
        final_test_metrics,
    )

    save_final_metrics(
        final_test_metrics,
        result_directory / "final_metrics.csv",
    )

    print("\nSaved files:")
    print(final_checkpoint_path)

    print(result_directory / "training_history.csv")

    print(result_directory / "final_metrics.csv")

    print(result_directory / "config.json")


if __name__ == "__main__":
    main()