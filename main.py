import logging
import os
import gc
import argparse
import math
import random
import warnings

# 先导入 PyTorch，避免 Windows DLL 加载冲突
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 导入 STGCN 项目自己的代码
from script import dataloader, utility, earlystopping, opt
from model import models

# import nni


def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description="STGCN")
    parser.add_argument(
        "--enable_cuda", type=bool, default=True, help="enable CUDA, default as True"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="set the random seed for stabilizing experiment results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="metr-la",
        choices=["metr-la", "pems-bay", "pemsd7-m"],
    )
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument(
        "--n_pred",
        type=int,
        default=3,
        help="the number of time interval for predcition, default as 3",
    )
    parser.add_argument("--time_intvl", type=int, default=5)
    parser.add_argument("--Kt", type=int, default=3)
    parser.add_argument("--stblock_num", type=int, default=2)
    parser.add_argument("--act_func", type=str, default="glu", choices=["glu", "gtu"])
    parser.add_argument("--Ks", type=int, default=3, choices=[3, 2])
    parser.add_argument(
        "--graph_conv_type",
        type=str,
        default="cheb_graph_conv",
        choices=["cheb_graph_conv", "graph_conv"],
    )
    parser.add_argument(
        "--gso_type",
        type=str,
        default="sym_norm_lap",
        choices=["sym_norm_lap", "rw_norm_lap", "sym_renorm_adj", "rw_renorm_adj"],
    )
    parser.add_argument(
        "--enable_bias", type=bool, default=True, help="default as True"
    )
    parser.add_argument("--droprate", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--weight_decay_rate",
        type=float,
        default=0.001,
        help="weight decay (L2 penalty)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--epochs", type=int, default=1000, help="epochs, default as 1000"
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        choices=["adamw", "nadamw", "lion"],
        help="optimizer, default as nadamw",
    )
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    args = parser.parse_args()
    print("Training configs: {}".format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # Clean cache
    else:
        device = torch.device("cpu")
        gc.collect()  # Clean cache

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    return args, device, blocks


def data_preparate(args, device):
    """
    使用论文对齐的数据处理方式准备训练、验证和测试集。
    """

    # =====================================================
    # 1. 准备图结构
    # =====================================================

    adj, n_vertex = dataloader.load_adj(args.dataset)

    gso = utility.calc_gso(
        adj,
        args.gso_type,
    )

    if args.graph_conv_type == "cheb_graph_conv":
        gso = utility.calc_chebynet_gso(gso)

    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)

    args.gso = torch.from_numpy(gso).to(device)

    # =====================================================
    # 2. 使用论文式数据处理
    # =====================================================

    (
        zscore,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = dataloader.prepare_paper_data(
        dataset_name=args.dataset,
        n_his=args.n_his,
        n_pred=args.n_pred,
        device=device,
        n_train_days=34,
        n_val_days=5,
        n_test_days=5,
        day_slot=288,
    )

    # =====================================================
    # 3. 建立 PyTorch DataLoader
    # =====================================================

    train_data = utils.data.TensorDataset(
        x_train,
        y_train,
    )

    train_iter = utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    val_data = utils.data.TensorDataset(
        x_val,
        y_val,
    )

    val_iter = utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_data = utils.data.TensorDataset(
        x_test,
        y_test,
    )

    test_iter = utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # =====================================================
    # 4. 显示数据处理结果
    # =====================================================

    print("\nPaper-aligned data preparation:")

    print(f"Global mean: {zscore.mean:.6f}")

    print(f"Global std:  {zscore.std:.6f}")

    print(
        "Train X / y:",
        tuple(x_train.shape),
        tuple(y_train.shape),
    )

    print(
        "Val X / y:",
        tuple(x_val.shape),
        tuple(y_val.shape),
    )

    print(
        "Test X / y:",
        tuple(x_test.shape),
        tuple(y_test.shape),
    )

    return (
        n_vertex,
        zscore,
        train_iter,
        val_iter,
        test_iter,
    )


def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(
        delta=0.0,
        patience=args.patience,
        verbose=True,
        path="STGCN_" + args.dataset + ".pt",
    )

    if args.graph_conv_type == "cheb_graph_conv":
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate
        )
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay_rate,
            decoupled_weight_decay=True,
        )
    elif args.opt == "lion":
        optimizer = opt.Lion(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate
        )
    else:
        raise ValueError(f"ERROR: The {args.opt} optimizer is undefined.")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    return loss, es, model, optimizer, scheduler


def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = (
            torch.cuda.max_memory_allocated() / 1000000
            if torch.cuda.is_available()
            else 0
        )
        print(
            "Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB".format(
                epoch + 1,
                optimizer.param_groups[0]["lr"],
                l_sum / n,
                val_loss,
                gpu_mem_alloc,
            )
        )

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break


@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(
        f"Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}"
    )


if __name__ == "__main__":
    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
    test(zscore, loss, model, test_iter, args)
