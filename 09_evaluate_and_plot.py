import math
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore", category=UserWarning)

# Windows下先导入torch
import torch
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
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
BATCH_SIZE = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 1. 准备图结构
# =========================================================

adj, n_vertex = dataloader.load_adj(DATASET)

gso_sparse = utility.calc_gso(
    adj,
    "sym_norm_lap",
)

gso_sparse = utility.calc_chebynet_gso(
    gso_sparse
)

gso = gso_sparse.toarray().astype(np.float32)
gso = torch.from_numpy(gso).to(device)


# =========================================================
# 2. 创建模型并加载已训练参数
# =========================================================

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

blocks = [
    [1],
    [64, 16, 64],
    [64, 16, 64],
    [128, 128],
    [1],
]

model = models.STGCNChebGraphConv(
    args,
    blocks,
    n_vertex,
).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()


# =========================================================
# 3. 按照仓库方式划分和标准化数据
# =========================================================

data = pd.read_csv(DATA_PATH).to_numpy(
    dtype=np.float32
)

data_length = len(data)

len_val = int(math.floor(data_length * 0.15))
len_test = int(math.floor(data_length * 0.15))
len_train = data_length - len_val - len_test

train_data = data[:len_train]
test_data = data[len_train + len_val:]

scaler = StandardScaler()
scaler.fit(train_data)

test_normalized = scaler.transform(
    test_data
).astype(np.float32)

x_test, y_test = dataloader.data_transform(
    test_normalized,
    N_HIS,
    N_PRED,
    device,
)

test_dataset = data_utils.TensorDataset(
    x_test,
    y_test,
)

test_loader = data_utils.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================================================
# 4. 对完整测试集进行预测
# =========================================================

predictions_normalized = []
targets_normalized = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        prediction_batch = model(x_batch).view(
            len(x_batch),
            n_vertex,
        )

        predictions_normalized.append(
            prediction_batch.cpu().numpy()
        )

        targets_normalized.append(
            y_batch.cpu().numpy()
        )

predictions_normalized = np.concatenate(
    predictions_normalized,
    axis=0,
)

targets_normalized = np.concatenate(
    targets_normalized,
    axis=0,
)

# 恢复成真实交通速度
predictions = scaler.inverse_transform(
    predictions_normalized
)

targets = scaler.inverse_transform(
    targets_normalized
)


# =========================================================
# 5. 计算完整测试集指标
# =========================================================

absolute_error = np.abs(
    predictions - targets
)

squared_error = (
    predictions - targets
) ** 2

mae = absolute_error.mean()
rmse = np.sqrt(squared_error.mean())

# 防止真实值为0时出现除零错误
epsilon = 1e-6

mape = np.mean(
    absolute_error /
    np.maximum(np.abs(targets), epsilon)
) * 100

wmape = (
    absolute_error.sum() /
    np.maximum(np.abs(targets).sum(), epsilon)
) * 100


print("=" * 65)
print("1. 完整测试集")
print("=" * 65)

print("测试样本数量:", len(predictions))
print("每个样本预测节点数:", n_vertex)
print("预测总数:", predictions.size)
print("预测形状:", predictions.shape)
print("真实值形状:", targets.shape)


print("\n" + "=" * 65)
print("2. 完整测试指标")
print("=" * 65)

print(f"MAE:   {mae:.6f}")
print(f"RMSE:  {rmse:.6f}")
print(f"MAPE:  {mape:.4f}%")
print(f"WMAPE: {wmape:.4f}%")


# =========================================================
# 6. 计算每个传感器的MAE
# =========================================================

sensor_mae = absolute_error.mean(axis=0)

sorted_sensor_indices = np.argsort(sensor_mae)

best_sensor = int(sorted_sensor_indices[0])
worst_sensor = int(sorted_sensor_indices[-1])
median_sensor = int(
    sorted_sensor_indices[len(sorted_sensor_indices) // 2]
)

print("\n" + "=" * 65)
print("3. 每个传感器的表现")
print("=" * 65)

print(
    f"表现最好传感器: {best_sensor}, "
    f"MAE={sensor_mae[best_sensor]:.4f}"
)

print(
    f"中等表现传感器: {median_sensor}, "
    f"MAE={sensor_mae[median_sensor]:.4f}"
)

print(
    f"表现最差传感器: {worst_sensor}, "
    f"MAE={sensor_mae[worst_sensor]:.4f}"
)


# =========================================================
# 7. 保存指标
# =========================================================

metric_table = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "MAPE_percent",
            "WMAPE_percent",
        ],
        "Value": [
            mae,
            rmse,
            mape,
            wmape,
        ],
    }
)

metric_table.to_csv(
    "results_15min_metrics.csv",
    index=False,
)


# =========================================================
# 8. 绘制一天左右的真实值和预测值
# =========================================================

plot_length = min(288, len(predictions))

plot_table = pd.DataFrame(
    {
        "time_step": np.arange(plot_length),
        "true_speed": targets[
            :plot_length,
            median_sensor,
        ],
        "predicted_speed": predictions[
            :plot_length,
            median_sensor,
        ],
    }
)

plot_table.to_csv(
    "results_15min_predictions.csv",
    index=False,
)

plt.figure(figsize=(12, 5))

plt.plot(
    plot_table["time_step"],
    plot_table["true_speed"],
    label="True speed",
)

plt.plot(
    plot_table["time_step"],
    plot_table["predicted_speed"],
    label="Predicted speed",
)

plt.xlabel("5-minute time step")
plt.ylabel("Traffic speed")

plt.title(
    f"STGCN 15-minute prediction — "
    f"Sensor {median_sensor}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results_15min_prediction.png",
    dpi=200,
)

plt.show()

print("\n结果已经保存:")
print("results_15min_metrics.csv")
print("results_15min_predictions.csv")
print("results_15min_prediction.png")