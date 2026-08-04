import math
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore", category=UserWarning)

# Windows下先导入torch
import torch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model import models
from script import dataloader, utility


DATASET = "pemsd7-m"
DATA_PATH = Path("data/pemsd7-m/vel.csv")
MODEL_PATH = Path("STGCN_pemsd7-m.pt")

N_HIS = 12
N_PRED = 3
KT = 3
KS = 3
STBLOCK_NUM = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


adj, n_vertex = dataloader.load_adj(DATASET)

gso_sparse = utility.calc_gso(adj, "sym_norm_lap")

gso_sparse = utility.calc_chebynet_gso(gso_sparse)

gso = gso_sparse.toarray().astype(np.float32)
gso = torch.from_numpy(gso).to(device)


args = SimpleNamespace(
    Kt=KT,
    Ks=KS,
    n_his=N_HIS,
    act_func="glu",
    graph_conv_type="cheb_graph_conv",
    gso=gso,
    enable_bias=True,
    droprate=0.5,
)

# Keep the block settings aligned with baseline/main.py.
blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]]

model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"没有找到模型文件：{MODEL_PATH}")

try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    # 兼容不支持weights_only参数的PyTorch版本
    state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)
model.eval()


data = pd.read_csv(DATA_PATH).to_numpy(dtype=np.float32)

data_length = len(data)

len_val = int(math.floor(data_length * 0.15))
len_test = int(math.floor(data_length * 0.15))
len_train = data_length - len_val - len_test

train_data = data[:len_train]
test_data = data[len_train + len_val :]

zscore = StandardScaler()
zscore.fit(train_data)

test_normalized = zscore.transform(test_data).astype(np.float32)

# 第一个测试样本
x_numpy = test_normalized[0:N_HIS]

target_index = N_HIS + N_PRED - 1
y_numpy = test_normalized[target_index]

x = torch.tensor(x_numpy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

y = torch.tensor(y_numpy, dtype=torch.float32).unsqueeze(0).to(device)


with torch.no_grad():
    block1_output = model.st_blocks[0](x)

    block2_output = model.st_blocks[1](block1_output)

    output_block_result = model.output(block2_output)

    full_output = model(x)

    prediction_normalized = full_output.view(1, n_vertex)


prediction_original = zscore.inverse_transform(prediction_normalized.cpu().numpy())

target_original = zscore.inverse_transform(y.cpu().numpy())


print("=" * 68)
print("1. 完整STGCN结构")
print("=" * 68)

print(model)

print("\n剩余时间卷积长度Ko:", model.Ko)
print("ST-Conv Block数量:", len(model.st_blocks))


print("\n" + "=" * 68)
print("2. 完整模型形状变化")
print("=" * 68)

print("原始输入:          ", x.shape)
print("第一个Block输出:   ", block1_output.shape)
print("第二个Block输出:   ", block2_output.shape)
print("Output Block输出:  ", output_block_result.shape)
print("完整模型输出:      ", full_output.shape)
print("展平后预测形状:    ", prediction_normalized.shape)


print("\n时间长度变化:")
print("12 → 8 → 4 → 1")

print("\n通道数量变化:")
print("1 → 64 → 64 → 1")


print("\n" + "=" * 68)
print("3. 分步骤和完整forward比较")
print("=" * 68)

print(
    "两种执行结果是否一致:", torch.allclose(output_block_result, full_output, atol=1e-6)
)


print("\n" + "=" * 68)
print("4. 第一个真实测试样本")
print("=" * 68)

print("输入形状:", x.shape)
print("预测目标形状:", y.shape)
print("预测设备:", full_output.device)

print("\n前5个传感器:")
for sensor_index in range(5):
    prediction = prediction_original[0, sensor_index]

    target = target_original[0, sensor_index]

    error = abs(prediction - target)

    print(
        f"传感器{sensor_index}: "
        f"预测={prediction:.2f}, "
        f"真实={target:.2f}, "
        f"绝对误差={error:.2f}"
    )


print("\n" + "=" * 68)
print("5. 模型参数数量")
print("=" * 68)

total_parameters = sum(parameter.numel() for parameter in model.parameters())

trainable_parameters = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)

print("总参数数量:", total_parameters)
print("可训练参数数量:", trainable_parameters)
