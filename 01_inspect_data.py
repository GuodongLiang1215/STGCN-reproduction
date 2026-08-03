from pathlib import Path

import pandas as pd
import scipy.sparse as sp


DATA_DIR = Path("data/pemsd7-m")

# 每一行代表一个时间点，每一列代表一个交通传感器
vel = pd.read_csv(DATA_DIR / "vel.csv")

# 228个交通传感器之间的图连接关系
adj = sp.load_npz(DATA_DIR / "adj.npz")

print("=" * 50)
print("1. 交通速度数据")
print("=" * 50)
print("速度数据形状:", vel.shape)
print("时间点数量:", vel.shape[0])
print("传感器数量:", vel.shape[1])

print("\n前5个时间点、前5个传感器的数据:")
print(vel.iloc[:5, :5])

print("\n速度最小值:", vel.to_numpy().min())
print("速度最大值:", vel.to_numpy().max())
print("速度平均值:", vel.to_numpy().mean())

print("\n" + "=" * 50)
print("2. 图邻接矩阵")
print("=" * 50)
print("邻接矩阵形状:", adj.shape)
print("非零元素数量:", adj.nnz)
print("矩阵类型:", type(adj))

print("\n前5行、前5列:")
print(adj[:5, :5].toarray())