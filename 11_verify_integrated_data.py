# Windows 下首先导入 torch
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
    x_val,
    y_val,
    x_test,
    y_test,
) = dataloader.prepare_paper_data(
    dataset_name="pemsd7-m",
    n_his=12,
    n_pred=3,
    device=device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
)

print("=" * 65)
print("正式 dataloader 集成检查")
print("=" * 65)

print("运行设备:", device)

print("\n全局均值:", scaler.mean)
print("全局标准差:", scaler.std)

print("\n训练输入:", x_train.shape)
print("训练目标:", y_train.shape)

print("\n验证输入:", x_val.shape)
print("验证目标:", y_val.shape)

print("\n测试输入:", x_test.shape)
print("测试目标:", y_test.shape)

print("\n训练输入所在设备:", x_train.device)
print("训练目标所在设备:", y_train.device)

# 检查反标准化
normalized_value = x_train[
    0,
    0,
    0,
    0,
].item()

restored_value = scaler.inverse_transform(
    normalized_value
)

print("\n第一个标准化数值:", normalized_value)
print("恢复后的原始速度:", restored_value)

print("\n预期第一个原始速度约为: 71.1")