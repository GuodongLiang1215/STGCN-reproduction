from pathlib import Path

# Windows下先导入PyTorch，避免DLL加载冲突
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/pemsd7-m/vel.csv")

N_HIS = 12
N_PRED = 3
BATCH_SIZE = 32

# 与当前仓库保持相同的读取方式
data = pd.read_csv(DATA_PATH).to_numpy(dtype=np.float32)

# 使用前70%作为训练数据
train_end = int(len(data) * 0.7)
train_data = data[:train_end]

# 只使用训练集拟合标准化器
scaler = StandardScaler()
train_normalized = scaler.fit_transform(train_data).astype(np.float32)


def make_sample(data_array, start, n_his, n_pred):
    """从连续时间序列中制作一个STGCN样本。"""

    end = start + n_his

    # 过去12个时间点
    x = data_array[start:end]

    # 15分钟后的目标
    target_index = end + n_pred - 1
    y = data_array[target_index]

    return x, y


print("=" * 60)
print("1. 单个样本")
print("=" * 60)

x, y = make_sample(
    train_normalized,
    start=0,
    n_his=N_HIS,
    n_pred=N_PRED,
)

print("NumPy输入形状:", x.shape)
print("NumPy目标形状:", y.shape)

# 转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 增加batch和channel两个维度
x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
y_tensor = y_tensor.unsqueeze(0)

print("\n增加维度后:")
print("输入张量形状:", x_tensor.shape)
print("目标张量形状:", y_tensor.shape)

print("\n各维度含义:")
print("输入：[样本数量, 特征数量, 时间点数量, 节点数量]")
print("输入：[    1,        1,         12,        228]")
print("目标：[样本数量, 节点数量]")


print("\n" + "=" * 60)
print("2. 构造一个包含32个样本的batch")
print("=" * 60)

batch_x = []
batch_y = []

for start in range(BATCH_SIZE):
    sample_x, sample_y = make_sample(
        train_normalized,
        start=start,
        n_his=N_HIS,
        n_pred=N_PRED,
    )

    batch_x.append(sample_x)
    batch_y.append(sample_y)

batch_x = np.stack(batch_x)
batch_y = np.stack(batch_y)

batch_x = torch.tensor(batch_x, dtype=torch.float32).unsqueeze(1)
batch_y = torch.tensor(batch_y, dtype=torch.float32)

print("Batch输入形状:", batch_x.shape)
print("Batch目标形状:", batch_y.shape)


print("\n" + "=" * 60)
print("3. 放入GPU")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

print("使用设备:", device)
print("输入所在设备:", batch_x.device)
print("目标所在设备:", batch_y.device)