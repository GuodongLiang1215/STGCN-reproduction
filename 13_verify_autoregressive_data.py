# Windows 下优先导入 torch
import torch

from script import dataloader


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


(
    scaler,
    x_train,
    y_train,
    val_sequences,
    test_sequences,
) = dataloader.prepare_autoregressive_data(
    dataset_name="pemsd7-m",
    n_his=12,
    max_pred_steps=9,
    device=device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
)


print("=" * 70)
print("正式递归多步数据集成检查")
print("=" * 70)

print("运行设备:", device)

print("\n全局均值:", scaler.mean)
print("全局标准差:", scaler.std)


print("\n" + "=" * 70)
print("1. 一步预测训练数据")
print("=" * 70)

print("训练输入 X:", x_train.shape)
print("训练标签 y:", y_train.shape)

print("\n训练输入含义:")
print("[样本数, 特征数, 12个历史时间点, 228个节点]")

print("\n训练标签含义:")
print("[样本数, 228个节点]")
print("标签对应历史窗口之后的5分钟")


print("\n" + "=" * 70)
print("2. 完整验证和测试序列")
print("=" * 70)

print("验证完整序列:", val_sequences.shape)
print("测试完整序列:", test_sequences.shape)

print("\n完整序列含义:")
print("[样本数, 21个时间点, 228个节点, 1个特征]")


print("\n" + "=" * 70)
print("3. 张量设备")
print("=" * 70)

print("训练输入设备:", x_train.device)
print("训练标签设备:", y_train.device)
print("验证序列设备:", val_sequences.device)
print("测试序列设备:", test_sequences.device)


print("\n" + "=" * 70)
print("4. 第一个训练标签检查")
print("=" * 70)

first_normalized_target = y_train[
    0,
    0,
].item()

first_original_target = scaler.inverse_transform(
    first_normalized_target
)

print("第一个标准化训练标签:", first_normalized_target)
print("恢复后的原始速度:", first_original_target)
print("预期5分钟后的速度约为: 69.2")


print("\n" + "=" * 70)
print("5. 第一个测试序列的目标位置")
print("=" * 70)

for step in (3, 6, 9):
    sequence_index = 12 + step - 1

    normalized_value = test_sequences[
        0,
        sequence_index,
        0,
        0,
    ].item()

    original_value = scaler.inverse_transform(
        normalized_value
    )

    minutes = step * 5

    print(
        f"{minutes}分钟目标："
        f"完整序列索引={sequence_index}, "
        f"传感器0真实速度={original_value:.2f}"
    )


print("\n" + "=" * 70)
print("6. 结论")
print("=" * 70)

print("训练：使用过去12步预测下一步")
print("验证和测试：保留未来9步真实答案")
print("后续模型将递归运行9次")
print("评估第3、6、9步")