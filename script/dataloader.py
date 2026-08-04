# Windows 下优先导入 torch，避免 DLL 加载顺序问题
import torch

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

# =========================================================
# 1. 全局标准化器
# =========================================================


@dataclass
class GlobalStandardScaler:
    """
    使用一个全局均值和一个全局标准差进行标准化。

    这与 sklearn 的 StandardScaler 不同：
    StandardScaler 会为每个传感器分别计算均值和标准差；
    本类只保存一组 mean 和 std。
    """

    mean: float
    std: float

    def transform(self, data):
        """把原始速度转换成标准化数值。"""

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """把标准化数值恢复成真实交通速度。"""

        return data * self.std + self.mean


# =========================================================
# 2. 加载图邻接矩阵
# =========================================================


def load_adj(dataset_name):
    """
    读取交通传感器之间的邻接矩阵。
    """

    dataset_path = os.path.join(
        "./data",
        dataset_name,
    )

    adj = sp.load_npz(os.path.join(dataset_path, "adj.npz"))

    adj = adj.tocsc()

    if dataset_name == "metr-la":
        n_vertex = 207
    elif dataset_name == "pems-bay":
        n_vertex = 325
    elif dataset_name == "pemsd7-m":
        n_vertex = 228
    else:
        raise ValueError(f"不支持的数据集：{dataset_name}")

    return adj, n_vertex


# =========================================================
# 3. 正确读取速度数据
# =========================================================


def load_velocity_data(dataset_name):
    """
    读取交通速度 CSV。

    header=None 很重要：
    原始 CSV 没有列名，第一行也是交通速度数据。
    """

    dataset_path = os.path.join(
        "./data",
        dataset_name,
    )

    file_path = os.path.join(
        dataset_path,
        "vel.csv",
    )

    velocity = pd.read_csv(
        file_path,
        header=None,
        dtype=np.float32,
    ).to_numpy()

    return velocity


# =========================================================
# 4. 按天生成完整序列
# =========================================================


def generate_daily_sequences(
    data,
    start_day,
    num_days,
    n_his,
    n_pred,
    day_slot=288,
):
    """
    按天制作滑动窗口。

    每个完整序列包括：
    - 过去 n_his 个时间点；
    - 直到预测目标所需的 n_pred 个间隔。

    输出形状：
    [样本数, 完整序列长度, 节点数, 特征数]
    """

    n_frame = n_his + n_pred
    n_vertex = data.shape[1]

    # 一个窗口每天可以摆放多少次
    slots_per_day = day_slot - n_frame + 1

    if slots_per_day <= 0:
        raise ValueError("n_his + n_pred 不能大于一天的时间点数量。")

    total_samples = num_days * slots_per_day

    sequences = np.empty(
        (
            total_samples,
            n_frame,
            n_vertex,
            1,
        ),
        dtype=np.float32,
    )

    sample_index = 0

    for day in range(
        start_day,
        start_day + num_days,
    ):
        day_start = day * day_slot

        for slot in range(slots_per_day):
            start = day_start + slot
            end = start + n_frame

            sequences[
                sample_index,
                :,
                :,
                0,
            ] = data[start:end]

            sample_index += 1

    return sequences


# =========================================================
# 5. 把完整序列拆成 X 和 y
# =========================================================


def split_sequences(
    sequences,
    n_his,
    n_pred,
):
    """
    把完整序列拆成模型输入 X 和预测目标 y。

    X 输出形状：
    [样本数, 特征数, 历史时间点数, 节点数]

    y 输出形状：
    [样本数, 节点数]
    """

    target_index = n_his + n_pred - 1

    # 前 n_his 个时间点作为模型输入
    x = sequences[:, :n_his, :, :]

    # 第 target_index 个时间点作为预测答案
    y = sequences[
        :,
        target_index,
        :,
        0,
    ]

    # 原来：
    # [样本, 时间, 节点, 特征]
    #
    # 改为 PyTorch 模型需要的：
    # [样本, 特征, 时间, 节点]
    x = np.transpose(
        x,
        (0, 3, 1, 2),
    )

    x = np.ascontiguousarray(
        x,
        dtype=np.float32,
    )

    y = np.ascontiguousarray(
        y,
        dtype=np.float32,
    )

    return x, y


# =========================================================
# 6. 论文式完整数据准备
# =========================================================


def prepare_paper_data(
    dataset_name,
    n_his,
    n_pred,
    device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
):
    """
    使用原论文的数据处理方式：

    1. CSV 使用 header=None；
    2. 34 天训练、5 天验证、5 天测试；
    3. 每天单独制作窗口；
    4. 使用训练序列计算一个全局均值和标准差；
    5. 转换成 PyTorch 张量。
    """

    raw_data = load_velocity_data(dataset_name)

    expected_points = (n_train_days + n_val_days + n_test_days) * day_slot

    if len(raw_data) != expected_points:
        raise ValueError(
            "数据时间点数量与论文配置不一致。"
            f"当前为 {len(raw_data)}，"
            f"预期为 {expected_points}。"
        )

    # ---------------------------------------------
    # 按天制作训练、验证和测试序列
    # ---------------------------------------------

    train_sequences = generate_daily_sequences(
        raw_data,
        start_day=0,
        num_days=n_train_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )

    val_sequences = generate_daily_sequences(
        raw_data,
        start_day=n_train_days,
        num_days=n_val_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )

    test_sequences = generate_daily_sequences(
        raw_data,
        start_day=n_train_days + n_val_days,
        num_days=n_test_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )

    # ---------------------------------------------
    # 计算论文式全局均值和标准差
    # ---------------------------------------------

    global_mean = float(
        np.mean(
            train_sequences,
            dtype=np.float64,
        )
    )

    global_std = float(
        np.std(
            train_sequences,
            dtype=np.float64,
        )
    )

    if global_std == 0:
        raise ValueError("训练序列的标准差为 0，无法标准化。")

    scaler = GlobalStandardScaler(
        mean=global_mean,
        std=global_std,
    )

    # 同一组参数处理三个集合
    train_sequences = scaler.transform(train_sequences).astype(np.float32)

    val_sequences = scaler.transform(val_sequences).astype(np.float32)

    test_sequences = scaler.transform(test_sequences).astype(np.float32)

    # ---------------------------------------------
    # 拆分模型输入和目标
    # ---------------------------------------------

    x_train, y_train = split_sequences(
        train_sequences,
        n_his,
        n_pred,
    )

    x_val, y_val = split_sequences(
        val_sequences,
        n_his,
        n_pred,
    )

    x_test, y_test = split_sequences(
        test_sequences,
        n_his,
        n_pred,
    )

    # ---------------------------------------------
    # 转换成 PyTorch 张量并放入指定设备
    # ---------------------------------------------

    x_train = torch.from_numpy(x_train).to(device)

    y_train = torch.from_numpy(y_train).to(device)

    x_val = torch.from_numpy(x_val).to(device)

    y_val = torch.from_numpy(y_val).to(device)

    x_test = torch.from_numpy(x_test).to(device)

    y_test = torch.from_numpy(y_test).to(device)

    return (
        scaler,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


# =========================================================
# 7. 制作一步预测训练数据
# =========================================================


def split_one_step_training_data(
    sequences,
    n_his,
):
    """
    从完整多步序列中制作“一步预测”训练数据。

    输入：
    sequences:
    [样本数, 完整序列长度, 节点数, 特征数]

    模型输入：
    前 n_his 个时间点。

    训练标签：
    紧接在历史窗口之后的下一个时间点，
    也就是 5 分钟后的交通速度。
    """

    # 过去 12 个时间点
    x = sequences[
        :,
        0:n_his,
        :,
        :,
    ]

    # 索引 n_his 是历史窗口之后的第一个时刻
    # 当 n_his=12 时，这里就是索引12
    y = sequences[
        :,
        n_his,
        :,
        0,
    ]

    # 原始顺序：
    # [样本, 时间, 节点, 特征]
    #
    # PyTorch STGCN 需要：
    # [样本, 特征, 时间, 节点]
    x = np.transpose(
        x,
        (0, 3, 1, 2),
    )

    x = np.ascontiguousarray(
        x,
        dtype=np.float32,
    )

    y = np.ascontiguousarray(
        y,
        dtype=np.float32,
    )

    return x, y


# =========================================================
# 8. 原作者递归多步预测数据准备
# =========================================================


def prepare_autoregressive_data(
    dataset_name,
    n_his,
    max_pred_steps,
    device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
):
    """
    按原作者的递归多步预测方式准备数据。

    训练：
    使用过去 n_his 个时间点，
    预测紧接着的下一个时间点。

    验证和测试：
    保留包含未来 max_pred_steps 个时间点的完整序列，
    后续将模型递归使用多次。

    当：
    n_his = 12
    max_pred_steps = 9

    完整序列长度就是：
    12 + 9 = 21
    """

    raw_data = load_velocity_data(dataset_name)

    expected_points = (n_train_days + n_val_days + n_test_days) * day_slot

    if len(raw_data) != expected_points:
        raise ValueError(
            "数据时间点数量与论文配置不一致。"
            f"当前为 {len(raw_data)}，"
            f"预期为 {expected_points}。"
        )

    # -----------------------------------------------------
    # 1. 按天生成长度为 n_his + max_pred_steps 的完整序列
    # -----------------------------------------------------

    train_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=0,
        num_days=n_train_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )

    val_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=n_train_days,
        num_days=n_val_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )

    test_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=n_train_days + n_val_days,
        num_days=n_test_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )

    # -----------------------------------------------------
    # 2. 计算一个全局平均值和标准差
    # -----------------------------------------------------

    global_mean = float(
        np.mean(
            train_sequences,
            dtype=np.float64,
        )
    )

    global_std = float(
        np.std(
            train_sequences,
            dtype=np.float64,
        )
    )

    if global_std == 0:
        raise ValueError("训练序列标准差为 0，无法标准化。")

    scaler = GlobalStandardScaler(
        mean=global_mean,
        std=global_std,
    )

    # 三个集合使用同一套训练集统计参数
    train_sequences = scaler.transform(train_sequences).astype(np.float32)

    val_sequences = scaler.transform(val_sequences).astype(np.float32)

    test_sequences = scaler.transform(test_sequences).astype(np.float32)

    # -----------------------------------------------------
    # 3. 训练集只制作一步预测任务
    # -----------------------------------------------------

    x_train, y_train = split_one_step_training_data(
        sequences=train_sequences,
        n_his=n_his,
    )

    # -----------------------------------------------------
    # 4. 转换为 PyTorch 张量
    # -----------------------------------------------------

    x_train = torch.from_numpy(x_train).to(device)

    y_train = torch.from_numpy(y_train).to(device)

    # 验证集和测试集保留完整21步序列
    # 后续递归预测需要使用这些完整真实目标
    val_sequences = torch.from_numpy(
        np.ascontiguousarray(
            val_sequences,
            dtype=np.float32,
        )
    ).to(device)

    test_sequences = torch.from_numpy(
        np.ascontiguousarray(
            test_sequences,
            dtype=np.float32,
        )
    ).to(device)

    return (
        scaler,
        x_train,
        y_train,
        val_sequences,
        test_sequences,
    )
