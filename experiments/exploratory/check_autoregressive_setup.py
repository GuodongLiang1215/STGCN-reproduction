from pathlib import Path

# Windows 下优先导入 torch
import torch

import numpy as np

from script import dataloader


DATASET = "pemsd7-m"

N_HIS = 12

# 一次递归预测到未来9个时间间隔
# 每个间隔5分钟，因此最多预测45分钟
N_PRED = 9

DAY_SLOT = 288

N_TRAIN_DAYS = 34
N_VAL_DAYS = 5
N_TEST_DAYS = 5

# 一个完整序列需要：
# 12个历史点 + 9个未来点 = 21个时间点
N_FRAME = N_HIS + N_PRED


raw_data = dataloader.load_velocity_data(DATASET)

if raw_data.shape != (12672, 228):
    raise ValueError(f"数据形状错误：{raw_data.shape}")


train_sequences = dataloader.generate_daily_sequences(
    data=raw_data,
    start_day=0,
    num_days=N_TRAIN_DAYS,
    n_his=N_HIS,
    n_pred=N_PRED,
    day_slot=DAY_SLOT,
)

val_sequences = dataloader.generate_daily_sequences(
    data=raw_data,
    start_day=N_TRAIN_DAYS,
    num_days=N_VAL_DAYS,
    n_his=N_HIS,
    n_pred=N_PRED,
    day_slot=DAY_SLOT,
)

test_sequences = dataloader.generate_daily_sequences(
    data=raw_data,
    start_day=N_TRAIN_DAYS + N_VAL_DAYS,
    num_days=N_TEST_DAYS,
    n_his=N_HIS,
    n_pred=N_PRED,
    day_slot=DAY_SLOT,
)


global_mean = float(np.mean(train_sequences, dtype=np.float64))

global_std = float(np.std(train_sequences, dtype=np.float64))

train_normalized = (train_sequences - global_mean) / global_std


# 输入：索引0～11，即过去12个时间点
x_train = train_normalized[:, 0:N_HIS, :, :]

# 训练标签：索引12，即最后一个输入时刻的5分钟后
y_train = train_normalized[:, N_HIS, :, 0]

# 改成 PyTorch 模型需要的顺序：
# [样本, 特征, 时间, 节点]
x_train = np.transpose(x_train, (0, 3, 1, 2))

x_train_tensor = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))

y_train_tensor = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))


# 最后一个历史时刻是索引11
#
# 索引12：5分钟后
# 索引13：10分钟后
# 索引14：15分钟后
#
# 索引17：30分钟后
# 索引20：45分钟后

INDEX_5MIN = N_HIS
INDEX_15MIN = N_HIS + 3 - 1
INDEX_30MIN = N_HIS + 6 - 1
INDEX_45MIN = N_HIS + 9 - 1


slots_per_day = DAY_SLOT - N_FRAME + 1

print("=" * 70)
print("1. 原作者递归多步预测设置")
print("=" * 70)

print("历史时间点数量:", N_HIS)
print("最大预测步数:", N_PRED)
print("完整序列长度:", N_FRAME)

print("一天样本数量:", f"{DAY_SLOT} - {N_FRAME} + 1 = " f"{slots_per_day}")


print("\n" + "=" * 70)
print("2. 完整序列形状")
print("=" * 70)

print("训练完整序列:", train_sequences.shape)
print("验证完整序列:", val_sequences.shape)
print("测试完整序列:", test_sequences.shape)

print("\n预期样本数量:")
print("训练:", f"34 × {slots_per_day} = " f"{34 * slots_per_day}")
print("验证:", f"5 × {slots_per_day} = " f"{5 * slots_per_day}")
print("测试:", f"5 × {slots_per_day} = " f"{5 * slots_per_day}")


print("\n" + "=" * 70)
print("3. 全局标准化")
print("=" * 70)

print("全局均值:", global_mean)
print("全局标准差:", global_std)

print("标准化后训练序列均值:", float(train_normalized.mean()))

print("标准化后训练序列标准差:", float(train_normalized.std()))


print("\n" + "=" * 70)
print("4. 真正用于训练的输入和标签")
print("=" * 70)

print("训练输入 X:", x_train_tensor.shape)
print("训练标签 y:", y_train_tensor.shape)

print("\n训练输入使用完整序列索引: 0 到 11")
print("训练标签使用完整序列索引: 12")
print("也就是训练模型预测5分钟后，而不是直接预测45分钟后。")


print("\n" + "=" * 70)
print("5. 第一个样本的时间位置")
print("=" * 70)

print("过去12个历史点:")
print("索引0到11")

print("\n训练标签:")
print(f"索引{INDEX_5MIN}，" "对应5分钟后")

print("\n论文报告的三个预测尺度:")
print(f"索引{INDEX_15MIN}，" "对应递归第3步，即15分钟后")
print(f"索引{INDEX_30MIN}，" "对应递归第6步，即30分钟后")
print(f"索引{INDEX_45MIN}，" "对应递归第9步，即45分钟后")


print("\n" + "=" * 70)
print("6. 第一个样本中传感器0的真实速度")
print("=" * 70)

print("历史速度:")
print(raw_data[0:N_HIS, 0])

print("\n5分钟后的真实速度:", raw_data[INDEX_5MIN, 0])

print("15分钟后的真实速度:", raw_data[INDEX_15MIN, 0])

print("30分钟后的真实速度:", raw_data[INDEX_30MIN, 0])

print("45分钟后的真实速度:", raw_data[INDEX_45MIN, 0])


print("\n" + "=" * 70)
print("7. 递归预测过程")
print("=" * 70)

print("第1次:")
print("使用真实索引0～11 → 预测索引12")

print("\n第2次:")
print("使用真实索引1～11 + 预测索引12")
print("→ 预测索引13")

print("\n第3次:")
print("使用真实索引2～11 + 预测索引12、13")
print("→ 预测索引14，即15分钟后")

print("\n继续相同操作:")
print("第6次预测得到30分钟后")
print("第9次预测得到45分钟后")


print("\n" + "=" * 70)
print("8. 结论")
print("=" * 70)

print("训练任务：一步预测，也就是5分钟后")
print("测试任务：把模型递归使用9次")
print("最终评价：第3、6、9次预测")
print("分别对应15、30、45分钟")
