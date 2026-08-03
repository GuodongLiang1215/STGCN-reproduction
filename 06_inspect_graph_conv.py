# Windows中先导入torch，避免DLL加载顺序问题
import torch

import numpy as np

from script import dataloader, utility
from model.layers import GraphConvLayer


DATASET = "pemsd7-m"

BATCH_SIZE = 32
C_IN = 64
C_OUT = 16
TIME_LENGTH = 10
KS = 3

GRAPH_CONV_TYPE = "cheb_graph_conv"
GSO_TYPE = "sym_norm_lap"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# 1. 加载邻接矩阵，并制作图移位算子GSO
# ---------------------------------------------------------
adj, n_vertex = dataloader.load_adj(DATASET)

gso_sparse = utility.calc_gso(
    adj,
    GSO_TYPE,
)

# Chebyshev图卷积需要进一步缩放GSO
gso_cheb_sparse = utility.calc_chebynet_gso(
    gso_sparse,
)

gso = gso_cheb_sparse.toarray()
gso = gso.astype(np.float32)
gso = torch.from_numpy(gso).to(device)


print("=" * 65)
print("1. 图结构")
print("=" * 65)

print("传感器节点数量:", n_vertex)
print("原始邻接矩阵形状:", adj.shape)
print("GSO形状:", gso.shape)
print("GSO所在设备:", gso.device)

print("\nGSO前5行、前5列:")
print(gso[:5, :5])


# ---------------------------------------------------------
# 2. 模拟第一次时间卷积的输出
# ---------------------------------------------------------
x = torch.randn(
    BATCH_SIZE,
    C_IN,
    TIME_LENGTH,
    n_vertex,
    device=device,
)

print("\n" + "=" * 65)
print("2. 图卷积输入")
print("=" * 65)

print("输入形状:", x.shape)
print("含义: [batch, channel, time, node]")


# ---------------------------------------------------------
# 3. 创建图卷积层
# ---------------------------------------------------------
graph_conv = GraphConvLayer(
    graph_conv_type=GRAPH_CONV_TYPE,
    c_in=C_IN,
    c_out=C_OUT,
    Ks=KS,
    gso=gso,
    bias=True,
).to(device)

graph_conv.eval()


with torch.no_grad():
    # 先将通道数从64对齐到16
    x_aligned = graph_conv.align(x)

    # ChebGraphConv内部会把输入转换为：
    # [batch, time, node, channel]
    x_permuted = x_aligned.permute(0, 2, 3, 1)

    # T0：节点自身
    x_0 = x_permuted

    # T1：经过一次图传播
    x_1 = torch.einsum(
        "hi,btij->bthj",
        gso,
        x_permuted,
    )

    # T2：Chebyshev递推
    x_2 = (
        torch.einsum(
            "hi,btij->bthj",
            2 * gso,
            x_1,
        )
        - x_0
    )

    cheb_features = torch.stack(
        [x_0, x_1, x_2],
        dim=2,
    )

    # 图卷积内部原始输出
    raw_graph_output = graph_conv.cheb_graph_conv(
        x_aligned
    )

    # 完整图卷积层输出，包括残差连接
    output = graph_conv(x)


print("\n" + "=" * 65)
print("3. 通道对齐")
print("=" * 65)

print("对齐前形状:", x.shape)
print("对齐后形状:", x_aligned.shape)

print("\n为什么64变成16:")
print("STGCN采用瓶颈结构：64 → 16 → 64")


print("\n" + "=" * 65)
print("4. Chebyshev图特征")
print("=" * 65)

print("T0形状:", x_0.shape)
print("T1形状:", x_1.shape)
print("T2形状:", x_2.shape)
print("堆叠后形状:", cheb_features.shape)

print("\n三个Chebyshev项的含义:")
print("T0：节点自身的信息")
print("T1：经过一次图传播的信息")
print("T2：经过更远邻域传播的信息")


print("\n" + "=" * 65)
print("5. 图卷积权重和输出")
print("=" * 65)

print(
    "图卷积权重形状:",
    graph_conv.cheb_graph_conv.weight.shape
)

print(
    "图卷积内部输出形状:",
    raw_graph_output.shape
)

print(
    "完整图卷积层输出形状:",
    output.shape
)

print("\n最终含义:")
print("32个样本")
print("16个空间特征通道")
print("10个时间位置")
print("228个交通传感器")
print("运行设备:", output.device)