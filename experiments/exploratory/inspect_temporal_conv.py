# Windows下先导入torch，避免DLL加载顺序问题
import torch

from model.layers import TemporalConvLayer


BATCH_SIZE = 32
C_IN = 1
C_OUT = 64
N_HIS = 12
N_VERTEX = 228
KT = 3


# 模拟STGCN收到的一个batch
x = torch.randn(BATCH_SIZE, C_IN, N_HIS, N_VERTEX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device)

# 创建一个时间卷积层
temporal_conv = TemporalConvLayer(
    Kt=KT, c_in=C_IN, c_out=C_OUT, n_vertex=N_VERTEX, act_func="glu"
).to(device)

temporal_conv.eval()

print("=" * 60)
print("1. 时间卷积输入")
print("=" * 60)
print("输入形状:", x.shape)
print("卷积核大小:", (KT, 1))
print("输入通道数:", C_IN)
print("输出通道数:", C_OUT)


with torch.no_grad():
    # 查看时间卷积内部的原始卷积结果
    raw_conv = temporal_conv.causal_conv(x)

    # 完整的GLU时间卷积输出
    output = temporal_conv(x)


print("\n" + "=" * 60)
print("2. 卷积后的形状")
print("=" * 60)

print("原始卷积结果形状:", raw_conv.shape)
print("最终时间卷积输出:", output.shape)

print("\n为什么原始卷积有128个通道:")
print("2 × 输出通道数 =", 2 * C_OUT)
print("前64个通道用于内容P")
print("后64个通道用于门控Q")


print("\n" + "=" * 60)
print("3. GLU拆分")
print("=" * 60)

x_p = raw_conv[:, :C_OUT, :, :]
x_q = raw_conv[:, C_OUT:, :, :]
gate = torch.sigmoid(x_q)

print("内容部分P形状:", x_p.shape)
print("门控部分Q形状:", x_q.shape)
print("Sigmoid门形状:", gate.shape)

print("\n门控值最小值:", gate.min().item())
print("门控值最大值:", gate.max().item())
print("门控值平均值:", gate.mean().item())


print("\n" + "=" * 60)
print("4. 时间长度变化")
print("=" * 60)

expected_time = N_HIS - KT + 1

print("输入时间长度:", N_HIS)
print("时间卷积核大小:", KT)
print("输出时间长度:", expected_time)
print("计算公式: 12 - 3 + 1 = 10")


print("\n" + "=" * 60)
print("5. 最终含义")
print("=" * 60)

print("最终形状:", f"[{BATCH_SIZE}, {C_OUT}, {expected_time}, {N_VERTEX}]")
print("32个样本")
print("64个学习出来的时间特征")
print("10个新的时间位置")
print("228个交通传感器")
print("运行设备:", output.device)
