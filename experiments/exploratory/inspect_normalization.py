from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/pemsd7-m/vel.csv")

# 与当前仓库保持相同读取方式
data = pd.read_csv(DATA_PATH).to_numpy(dtype=np.float32)

# 模拟当前仓库的数据划分：70%训练、15%验证、15%测试
total_length = len(data)
train_end = int(total_length * 0.7)
val_end = int(total_length * 0.85)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

print("=" * 55)
print("1. 数据集划分")
print("=" * 55)
print("全部数据形状:", data.shape)
print("训练集形状:", train_data.shape)
print("验证集形状:", val_data.shape)
print("测试集形状:", test_data.shape)

# 只使用训练集计算均值和标准差
scaler = StandardScaler()
scaler.fit(train_data)

train_normalized = scaler.transform(train_data)

print("\n" + "=" * 55)
print("2. 原始速度与标准化结果")
print("=" * 55)

sensor_index = 0

print("传感器0训练集均值:", scaler.mean_[sensor_index])
print("传感器0训练集标准差:", scaler.scale_[sensor_index])

original_value = train_data[0, sensor_index]
normalized_value = train_normalized[0, sensor_index]

print("\n第一个原始速度:", original_value)
print("标准化后的速度:", normalized_value)

manual_result = (original_value - scaler.mean_[sensor_index]) / scaler.scale_[
    sensor_index
]

print("手动计算结果:", manual_result)

print("\n" + "=" * 55)
print("3. 标准化后整体情况")
print("=" * 55)
print("标准化训练集平均值:", train_normalized.mean())
print("标准化训练集标准差:", train_normalized.std())

# 恢复成原始速度
restored_value = (
    normalized_value * scaler.scale_[sensor_index] + scaler.mean_[sensor_index]
)

print("\n恢复后的速度:", restored_value)
