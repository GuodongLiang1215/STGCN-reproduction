import warnings

# 只为让本次检查输出更干净
warnings.filterwarnings("ignore", category=UserWarning)

# Windows下首先导入torch
import torch
import numpy as np

from script import dataloader, utility
from model.layers import STConvBlock


DATASET = "pemsd7-m"

BATCH_SIZE = 32
N_HIS = 12

KT = 3
KS = 3

CHANNELS = [64, 16, 64]

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
    gso_sparse,
)

gso = gso_sparse.toarray().astype(np.float32)
gso = torch.from_numpy(gso).to(device)


# =========================================================
# 2. 模拟原始STGCN输入
# =========================================================

x = torch.randn(
    BATCH_SIZE,
    1,
    N_HIS,
    n_vertex,
    device=device,
)


# =========================================================
# 3. 创建一个完整ST-Conv Block
# =========================================================

block = STConvBlock(
    Kt=KT,
    Ks=KS,
    n_vertex=n_vertex,
    last_block_channel=1,
    channels=CHANNELS,
    act_func="glu",
    graph_conv_type="cheb_graph_conv",
    gso=gso,
    bias=True,
    droprate=0.5,
).to(device)

# 检查阶段使用eval，避免Dropout随机影响结果
block.eval()


# =========================================================
# 4. 分步骤执行
# =========================================================

with torch.no_grad():
    # 第一次时间卷积
    x_t1 = block.tmp_conv1(x)

    # 图卷积
    x_g = block.graph_conv(x_t1)

    # 激活函数
    x_relu = block.relu(x_g)

    # 第二次时间卷积
    x_t2 = block.tmp_conv2(x_relu)

    # Layer Normalization
    x_ln = block.tc2_ln(
        x_t2.permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)

    # Dropout；eval模式下不会随机丢弃
    x_dropout = block.dropout(x_ln)

    # 完整Block的正式输出
    output = block(x)


print("=" * 65)
print("1. ST-Conv Block输入")
print("=" * 65)

print("输入形状:", x.shape)
print("通道设置:", CHANNELS)
print("结构: 时间卷积 → 图卷积 → 时间卷积")


print("\n" + "=" * 65)
print("2. 每一层的形状变化")
print("=" * 65)

print("原始输入:        ", x.shape)
print("第一次时间卷积:  ", x_t1.shape)
print("图卷积:          ", x_g.shape)
print("ReLU后:          ", x_relu.shape)
print("第二次时间卷积:  ", x_t2.shape)
print("LayerNorm后:     ", x_ln.shape)
print("Dropout后:       ", x_dropout.shape)
print("完整Block输出:   ", output.shape)


print("\n" + "=" * 65)
print("3. 时间长度变化")
print("=" * 65)

print("原始时间长度:", N_HIS)
print("第一次时间卷积: 12 - 3 + 1 = 10")
print("图卷积不改变时间长度: 10")
print("第二次时间卷积: 10 - 3 + 1 = 8")
print("一个ST-Conv Block最终时间长度: 8")


print("\n" + "=" * 65)
print("4. 通道变化")
print("=" * 65)

print("输入通道数: 1")
print("第一次时间卷积: 1 → 64")
print("图卷积瓶颈: 64 → 16")
print("第二次时间卷积: 16 → 64")


print("\n" + "=" * 65)
print("5. 最终结果")
print("=" * 65)

print("最终形状:", output.shape)
print("预期形状: [32, 64, 8, 228]")
print("运行设备:", output.device)

print(
    "手动分步骤结果与完整forward是否一致:",
    torch.allclose(x_dropout, output),
)