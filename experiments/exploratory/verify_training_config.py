import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Windows 下先导入 torch，避免 DLL 加载顺序问题
import torch
import numpy as np
from types import SimpleNamespace

from model import models
from script import dataloader, utility


DATASET = "pemsd7-m"

BATCH_SIZE = 50
N_HIS = 12
MAX_PRED_STEPS = 9

KT = 3
KS = 3

EPOCHS = 50
LEARNING_RATE = 0.001

LR_STEP_SIZE = 5
LR_GAMMA = 0.7


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
    # 原作者训练和测试时 keep_prob=1.0，
    # 相当于不使用 Dropout
    droprate=0.0,
)


# 原作者第一个Block：
# [输入1, 中间32, 输出64]
#
# 在当前PyTorch实现中，输入通道由前一个block单独传入，
# 因此 channels 写成：
# [时间卷积输出32, 图卷积输出32, 第二时间卷积输出64]
#
# 第二个Block同理：
# 输入64 → 时间32 → 图卷积32 → 输出128

blocks = [[1], [32, 32, 64], [32, 32, 128], [128, 128], [1]]


model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)

model.eval()


x = torch.randn(BATCH_SIZE, 1, N_HIS, n_vertex, device=device)

with torch.no_grad():
    block1_output = model.st_blocks[0](x)

    block2_output = model.st_blocks[1](block1_output)

    output_block_result = model.output(block2_output)

    full_output = model(x)


# TensorFlow 1 RMSPropOptimizer 的主要默认参数：
# decay/alpha = 0.9
# epsilon = 1e-10
# momentum = 0
#
# 原作者训练损失中没有真正加入weight decay，
# 因此这里设为0。

optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=LEARNING_RATE,
    alpha=0.9,
    eps=1e-10,
    momentum=0.0,
    weight_decay=0.0,
    centered=False,
)


# 原作者使用 tf.nn.l2_loss：
# 0.5 × 所有平方误差之和
#
# 它与 PyTorch 默认 MSELoss(mean) 不一样。

target = torch.randn(BATCH_SIZE, n_vertex, device=device)

prediction = full_output.view(BATCH_SIZE, n_vertex)

paper_l2_loss = 0.5 * torch.sum((prediction - target) ** 2)

mean_mse_loss = torch.mean((prediction - target) ** 2)


dropout_probabilities = [
    module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)
]


total_parameters = sum(parameter.numel() for parameter in model.parameters())

trainable_parameters = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)


print("=" * 72)
print("1. 论文训练配置")
print("=" * 72)

print("运行设备:", device)
print("数据集:", DATASET)
print("历史时间点:", N_HIS)
print("最大递归预测步数:", MAX_PRED_STEPS)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


print("\n" + "=" * 72)
print("2. 论文通道结构")
print("=" * 72)

print("PyTorch blocks:")
for index, block in enumerate(blocks):
    print(f"blocks[{index}] = {block}")

print("\n原作者结构:")
print("Block 1：1 → 32 → 32 → 64")
print("Block 2：64 → 32 → 32 → 128")


print("\n" + "=" * 72)
print("3. 完整模型形状")
print("=" * 72)

print("原始输入:", x.shape)
print("第一个Block:", block1_output.shape)
print("第二个Block:", block2_output.shape)
print("Output Block:", output_block_result.shape)
print("完整模型输出:", full_output.shape)

print("\n预期时间变化:")
print("12 → 8 → 4 → 1")

print("\n预期通道变化:")
print("1 → 64 → 128 → 1")

print(
    "\n分步骤与完整forward一致:",
    torch.allclose(output_block_result, full_output, atol=1e-6),
)


print("\n" + "=" * 72)
print("4. RMSProp设置")
print("=" * 72)

optimizer_group = optimizer.param_groups[0]

print("优化器类型:", type(optimizer).__name__)
print("初始学习率:", optimizer_group["lr"])
print("alpha:", optimizer_group["alpha"])
print("epsilon:", optimizer_group["eps"])
print("momentum:", optimizer_group["momentum"])
print("weight decay:", optimizer_group["weight_decay"])
print("centered:", optimizer_group["centered"])


print("\n" + "=" * 72)
print("5. 学习率计划")
print("=" * 72)

important_epochs = [
    1,
    5,
    6,
    10,
    11,
    15,
    16,
    20,
    21,
    25,
    26,
    30,
    31,
    35,
    36,
    40,
    41,
    45,
    46,
    50,
]

for epoch in important_epochs:
    completed_decay_steps = (epoch - 1) // LR_STEP_SIZE

    epoch_lr = LEARNING_RATE * LR_GAMMA**completed_decay_steps

    print(f"Epoch {epoch:02d}: " f"lr={epoch_lr:.10f}")


print("\n" + "=" * 72)
print("6. Loss对比")
print("=" * 72)

print("论文式L2 Loss:", paper_l2_loss.item())

print("PyTorch平均MSE:", mean_mse_loss.item())

print("\n二者数值不同是正常的。")
print("论文式Loss会把所有节点和样本的平方误差相加。")


print("\n" + "=" * 72)
print("7. Dropout检查")
print("=" * 72)

print("模型中所有Dropout概率:", dropout_probabilities)

print("预期全部为0.0")


print("\n" + "=" * 72)
print("8. 模型参数数量")
print("=" * 72)

print("总参数数量:", total_parameters)
print("可训练参数数量:", trainable_parameters)


print("\n" + "=" * 72)
print("9. 结论")
print("=" * 72)

print("论文通道结构已经能够在当前PyTorch模型中运行")
print("RMSProp参数已经准备完毕")
print("学习率每5轮乘以0.7")
print("Dropout已经关闭")
print("当前只做检查，没有训练和保存模型")
