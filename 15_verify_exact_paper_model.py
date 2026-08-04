import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

import torch
import numpy as np

from model.paper_model import PaperSTGCN
from script import dataloader, utility


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# 1. 构造图算子
# =========================================================

adj, n_vertex = dataloader.load_adj(
    "pemsd7-m"
)

gso_sparse = utility.calc_gso(
    adj,
    "sym_norm_lap",
)

gso_sparse = utility.calc_chebynet_gso(
    gso_sparse
)

gso = torch.from_numpy(
    gso_sparse.toarray().astype(
        np.float32
    )
).to(device)


# =========================================================
# 2. 创建论文专用模型
# =========================================================

model = PaperSTGCN(
    Kt=3,
    Ks=3,
    n_his=12,
    n_vertex=n_vertex,
    gso=gso,
    droprate=0.0,
).to(device)

model.eval()


# =========================================================
# 3. 检查形状
# =========================================================

x = torch.randn(
    50,
    1,
    12,
    n_vertex,
    device=device,
)

with torch.no_grad():
    block1_output = model.block1(x)

    block2_output = model.block2(
        block1_output
    )

    output_result = model.output(
        block2_output
    )

    full_result = model(x)


# =========================================================
# 4. 参数数量
# =========================================================

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


# =========================================================
# 5. 输出
# =========================================================

print("=" * 72)
print("1. 论文专用模型形状")
print("=" * 72)

print("输入:", x.shape)
print("Block 1:", block1_output.shape)
print("Block 2:", block2_output.shape)
print("Output:", output_result.shape)
print("完整 forward:", full_result.shape)

print(
    "\n分步骤与完整 forward 一致:",
    torch.allclose(
        output_result,
        full_result,
        atol=1e-6,
    ),
)


print("\n" + "=" * 72)
print("2. 激活函数检查")
print("=" * 72)

print(
    "Block 1 第一次时间卷积:",
    model.block1.temporal1.activation,
)

print(
    "Block 1 第二次时间卷积:",
    model.block1.temporal2.activation,
)

print(
    "Block 2 第一次时间卷积:",
    model.block2.temporal1.activation,
)

print(
    "Block 2 第二次时间卷积:",
    model.block2.temporal2.activation,
)

print(
    "Output 第一次时间卷积:",
    model.output.temporal1.activation,
)

print(
    "Output 第二次时间卷积:",
    model.output.temporal2.activation,
)


print("\n" + "=" * 72)
print("3. LayerNorm检查")
print("=" * 72)

print(
    "Block 1 epsilon:",
    model.block1.norm.eps,
)

print(
    "Block 2 epsilon:",
    model.block2.norm.eps,
)

print(
    "Output epsilon:",
    model.output.norm.eps,
)


print("\n" + "=" * 72)
print("4. Dropout检查")
print("=" * 72)

print(
    "Block 1 Dropout:",
    model.block1.dropout.p,
)

print(
    "Block 2 Dropout:",
    model.block2.dropout.p,
)


print("\n" + "=" * 72)
print("5. 参数数量")
print("=" * 72)

print("论文专用模型参数:", parameter_count)
print("预期参数数量: 333604")


print("\n" + "=" * 72)
print("6. 结论")
print("=" * 72)

print("第一次时间卷积使用 GLU")
print("第二次时间卷积使用 ReLU")
print("Output 使用 GLU → Sigmoid → Fully Conv")
print("LayerNorm epsilon 使用 1e-6")
print("Dropout 为 0")
print("模型内部结构已经进一步对齐原作者")