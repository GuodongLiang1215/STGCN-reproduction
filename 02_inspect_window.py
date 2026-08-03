from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/pemsd7-m/vel.csv")

# 与当前仓库训练代码保持相同的读取方式
vel = pd.read_csv(DATA_PATH)

# 仅用于检查CSV是否实际没有表头
vel_without_header = pd.read_csv(DATA_PATH, header=None)

print("=" * 55)
print("1. 两种CSV读取方式")
print("=" * 55)
print("仓库当前读取方式:", vel.shape)
print("按无表头方式读取:", vel_without_header.shape)

n_his = 12
n_pred = 3
sample_index = 0

# 第一个样本的输入
start = sample_index
end = start + n_his

x = vel.iloc[start:end].to_numpy(dtype=np.float32)

# 第一个样本的预测目标
target_index = end + n_pred - 1
y = vel.iloc[target_index].to_numpy(dtype=np.float32)

print("\n" + "=" * 55)
print("2. 第一个训练样本")
print("=" * 55)
print("输入使用的行:", start, "到", end - 1)
print("目标所在的行:", target_index)
print("输入X形状:", x.shape)
print("目标y形状:", y.shape)

print("\n传感器0过去12个时刻的速度:")
print(x[:, 0])

print("\n传感器0在15分钟后的真实速度:")
print(y[0])

print("\n前5个传感器在15分钟后的真实速度:")
print(y[:5])