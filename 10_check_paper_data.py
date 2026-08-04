from pathlib import Path

# Windows 下先导入 torch，避免之后可能出现 DLL 加载顺序问题
import torch

import numpy as np
import pandas as pd


# =========================================================
# 1. 论文实验配置
# =========================================================

DATA_PATH = Path("data/pemsd7-m/vel.csv")

N_ROUTE = 228          # 228 个交通传感器
DAY_SLOT = 288         # 一天有 288 个 5 分钟时间点

N_TRAIN_DAYS = 34
N_VAL_DAYS = 5
N_TEST_DAYS = 5

N_HIS = 12             # 使用过去 12 个时间点
N_PRED = 3             # 预测 3 个时间间隔以后，即 15 分钟后

# 一个完整序列包含：
# 12 个历史时刻 + 3 个预测间隔 = 15 个时间点
N_FRAME = N_HIS + N_PRED

# 目标位于序列中的最后一个时间点
TARGET_INDEX = N_HIS + N_PRED - 1


# =========================================================
# 2. 按“每天”制作滑动窗口
# =========================================================

def make_daily_sequences(
    data: np.ndarray,
    start_day: int,
    num_days: int,
) -> np.ndarray:
    """
    按天制作 STGCN 序列。

    输出形状：
    [样本数, 序列长度, 节点数, 特征数]
    """

    # 一个长度为 N_FRAME 的窗口在一天中可以放多少次
    slots_per_day = DAY_SLOT - N_FRAME + 1

    total_samples = num_days * slots_per_day

    sequences = np.empty(
        (
            total_samples,
            N_FRAME,
            N_ROUTE,
            1,
        ),
        dtype=np.float32,
    )

    sample_index = 0

    for day in range(start_day, start_day + num_days):
        day_start = day * DAY_SLOT

        for slot in range(slots_per_day):
            start = day_start + slot
            end = start + N_FRAME

            sequences[
                sample_index,
                :,
                :,
                0,
            ] = data[start:end]

            sample_index += 1

    return sequences


# =========================================================
# 3. 正确读取完整 CSV
# =========================================================

raw_data = pd.read_csv(
    DATA_PATH,
    header=None,
    dtype=np.float32,
).to_numpy()

if raw_data.shape != (12672, 228):
    raise ValueError(
        "数据形状不符合预期。"
        f"当前形状为 {raw_data.shape}，"
        "预期应为 (12672, 228)。"
    )


# =========================================================
# 4. 按 34 / 5 / 5 天生成序列
# =========================================================

train_sequences = make_daily_sequences(
    raw_data,
    start_day=0,
    num_days=N_TRAIN_DAYS,
)

val_sequences = make_daily_sequences(
    raw_data,
    start_day=N_TRAIN_DAYS,
    num_days=N_VAL_DAYS,
)

test_sequences = make_daily_sequences(
    raw_data,
    start_day=N_TRAIN_DAYS + N_VAL_DAYS,
    num_days=N_TEST_DAYS,
)


# =========================================================
# 5. 论文式全局 Z-score
# =========================================================

# 注意：
# 这里得到的是一个平均值和一个标准差，
# 不是 228 个传感器各自一套参数。
global_mean = float(
    np.mean(train_sequences, dtype=np.float64)
)

global_std = float(
    np.std(train_sequences, dtype=np.float64)
)

if global_std == 0:
    raise ValueError("训练数据标准差为 0，无法执行标准化。")

# 使用训练集得到的同一组均值和标准差
train_sequences -= global_mean
train_sequences /= global_std

val_sequences -= global_mean
val_sequences /= global_std

test_sequences -= global_mean
test_sequences /= global_std


# =========================================================
# 6. 拆分模型输入 X 和预测目标 y
# =========================================================

# 原始序列：
# [样本, 15个时间点, 228个节点, 1个特征]

# 输入只取前 12 个时间点
x_train = train_sequences[:, :N_HIS, :, :]
x_val = val_sequences[:, :N_HIS, :, :]
x_test = test_sequences[:, :N_HIS, :, :]

# 目标取第 14 号索引，即 15 分钟后的速度
y_train = train_sequences[:, TARGET_INDEX, :, 0]
y_val = val_sequences[:, TARGET_INDEX, :, 0]
y_test = test_sequences[:, TARGET_INDEX, :, 0]

# PyTorch STGCN 使用：
# [样本, 特征, 时间, 节点]
x_train = np.transpose(x_train, (0, 3, 1, 2))
x_val = np.transpose(x_val, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))

# 转换为 PyTorch 张量，只用于检查形状
x_train_tensor = torch.from_numpy(
    np.ascontiguousarray(x_train)
)

y_train_tensor = torch.from_numpy(
    np.ascontiguousarray(y_train)
)


# =========================================================
# 7. 输出检查信息
# =========================================================

slots_per_day = DAY_SLOT - N_FRAME + 1

print("=" * 68)
print("1. 原始数据")
print("=" * 68)

print("原始数据形状:", raw_data.shape)
print("总天数:", raw_data.shape[0] // DAY_SLOT)
print("每天时间点数量:", DAY_SLOT)
print("传感器数量:", N_ROUTE)


print("\n" + "=" * 68)
print("2. 论文数据划分")
print("=" * 68)

print("训练天数:", N_TRAIN_DAYS)
print("验证天数:", N_VAL_DAYS)
print("测试天数:", N_TEST_DAYS)
print("合计天数:", N_TRAIN_DAYS + N_VAL_DAYS + N_TEST_DAYS)


print("\n" + "=" * 68)
print("3. 每天单独制作窗口")
print("=" * 68)

print("历史时间点数量:", N_HIS)
print("预测间隔数量:", N_PRED)
print("完整序列长度:", N_FRAME)

print(
    "每天样本数:",
    f"{DAY_SLOT} - {N_FRAME} + 1 = {slots_per_day}",
)

print("训练序列形状:", train_sequences.shape)
print("验证序列形状:", val_sequences.shape)
print("测试序列形状:", test_sequences.shape)


print("\n" + "=" * 68)
print("4. 论文式全局标准化")
print("=" * 68)

print("全局训练集平均值:", global_mean)
print("全局训练集标准差:", global_std)

print(
    "标准化后训练序列平均值:",
    float(train_sequences.mean()),
)

print(
    "标准化后训练序列标准差:",
    float(train_sequences.std()),
)


print("\n" + "=" * 68)
print("5. 模型输入与目标")
print("=" * 68)

print("训练输入 X 形状:", x_train_tensor.shape)
print("训练目标 y 形状:", y_train_tensor.shape)

print("\nX 的维度含义:")
print("[样本数, 特征数, 历史时间点数, 节点数]")

print("\ny 的维度含义:")
print("[样本数, 节点数]")


print("\n" + "=" * 68)
print("6. 第一个样本")
print("=" * 68)

print("第一个样本使用原始索引: 0 到 14")
print("输入使用索引: 0 到 11")
print("目标使用索引:", TARGET_INDEX)

print("\n传感器 0 的前 12 个原始速度:")
print(raw_data[0:N_HIS, 0])

print("\n传感器 0 在 15 分钟后的真实速度:")
print(raw_data[TARGET_INDEX, 0])


print("\n" + "=" * 68)
print("7. 为什么窗口不能跨越午夜")
print("=" * 68)

last_start_day_0 = slots_per_day - 1
last_target_day_0 = last_start_day_0 + TARGET_INDEX

first_start_day_1 = DAY_SLOT
first_target_day_1 = first_start_day_1 + TARGET_INDEX

print("第 1 天最后一个窗口:")
print(
    f"开始索引={last_start_day_0}, "
    f"目标索引={last_target_day_0}"
)

print("第 2 天第一个窗口:")
print(
    f"开始索引={first_start_day_1}, "
    f"目标索引={first_target_day_1}"
)

print("\n第 1 天最后一个目标索引应为 287。")
print("第 2 天从索引 288 重新开始。")
print("因此没有任何一个样本同时包含两天的数据。")