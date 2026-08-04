# Import PyTorch before NumPy on Windows to avoid DLL load-order issues.
import torch

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class GlobalStandardScaler:
    """Normalize all sensors with one training-set mean and standard deviation."""

    mean: float
    std: float

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_adj(dataset_name):
    """Load the sparse sensor adjacency matrix for a supported dataset."""

    dataset_path = os.path.join("./data", dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, "adj.npz")).tocsc()

    if dataset_name == "metr-la":
        n_vertex = 207
    elif dataset_name == "pems-bay":
        n_vertex = 325
    elif dataset_name == "pemsd7-m":
        n_vertex = 228
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return adj, n_vertex


def load_velocity_data(dataset_name):
    """Load traffic speeds, preserving the first row as data rather than a header."""

    file_path = os.path.join("./data", dataset_name, "vel.csv")
    return pd.read_csv(file_path, header=None, dtype=np.float32).to_numpy()


def generate_daily_sequences(data, start_day, num_days, n_his, n_pred, day_slot=288):
    """Create `[sample, time, node, feature]` windows without crossing days."""

    n_frame = n_his + n_pred
    n_vertex = data.shape[1]
    slots_per_day = day_slot - n_frame + 1

    if slots_per_day <= 0:
        raise ValueError("n_his + n_pred cannot exceed the number of daily slots")

    sequences = np.empty(
        (num_days * slots_per_day, n_frame, n_vertex, 1), dtype=np.float32
    )
    sample_index = 0

    for day in range(start_day, start_day + num_days):
        day_start = day * day_slot
        for slot in range(slots_per_day):
            start = day_start + slot
            sequences[sample_index, :, :, 0] = data[start : start + n_frame]
            sample_index += 1

    return sequences


def split_sequences(sequences, n_his, n_pred):
    """Split full windows into model inputs and the requested forecast target."""

    target_index = n_his + n_pred - 1
    x = sequences[:, :n_his, :, :]
    y = sequences[:, target_index, :, 0]

    # Convert [sample, time, node, feature] to the model's [B, C, T, N].
    x = np.transpose(x, (0, 3, 1, 2))
    return (
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
    )


def _validate_length(raw_data, n_train_days, n_val_days, n_test_days, day_slot):
    expected_points = (n_train_days + n_val_days + n_test_days) * day_slot
    if len(raw_data) != expected_points:
        raise ValueError(
            "Dataset length does not match the configured day split: "
            f"found {len(raw_data)}, expected {expected_points}"
        )


def _fit_global_scaler(train_sequences):
    # The original code uses one scalar pair, not per-sensor statistics.
    global_mean = float(np.mean(train_sequences, dtype=np.float64))
    global_std = float(np.std(train_sequences, dtype=np.float64))
    if global_std == 0:
        raise ValueError("Training sequences have zero standard deviation")
    return GlobalStandardScaler(mean=global_mean, std=global_std)


def _normalize_splits(scaler, train_sequences, val_sequences, test_sequences):
    return tuple(
        scaler.transform(split).astype(np.float32)
        for split in (train_sequences, val_sequences, test_sequences)
    )


def prepare_paper_data(
    dataset_name,
    n_his,
    n_pred,
    device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
):
    """Prepare the original daily split for a fixed `n_pred` target horizon."""

    raw_data = load_velocity_data(dataset_name)
    _validate_length(raw_data, n_train_days, n_val_days, n_test_days, day_slot)

    train_sequences = generate_daily_sequences(
        raw_data,
        start_day=0,
        num_days=n_train_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )
    val_sequences = generate_daily_sequences(
        raw_data,
        start_day=n_train_days,
        num_days=n_val_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )
    test_sequences = generate_daily_sequences(
        raw_data,
        start_day=n_train_days + n_val_days,
        num_days=n_test_days,
        n_his=n_his,
        n_pred=n_pred,
        day_slot=day_slot,
    )

    scaler = _fit_global_scaler(train_sequences)
    train_sequences, val_sequences, test_sequences = _normalize_splits(
        scaler, train_sequences, val_sequences, test_sequences
    )

    x_train, y_train = split_sequences(train_sequences, n_his, n_pred)
    x_val, y_val = split_sequences(val_sequences, n_his, n_pred)
    x_test, y_test = split_sequences(test_sequences, n_his, n_pred)

    return (
        scaler,
        torch.from_numpy(x_train).to(device),
        torch.from_numpy(y_train).to(device),
        torch.from_numpy(x_val).to(device),
        torch.from_numpy(y_val).to(device),
        torch.from_numpy(x_test).to(device),
        torch.from_numpy(y_test).to(device),
    )


def split_one_step_training_data(sequences, n_his):
    """Build one-step targets while retaining full windows for recursive tests."""

    x = sequences[:, :n_his, :, :]
    y = sequences[:, n_his, :, 0]
    x = np.transpose(x, (0, 3, 1, 2))

    return (
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
    )


def prepare_autoregressive_data(
    dataset_name,
    n_his,
    max_pred_steps,
    device,
    n_train_days=34,
    n_val_days=5,
    n_test_days=5,
    day_slot=288,
):
    """Prepare one-step training data and full windows for recursive evaluation."""

    raw_data = load_velocity_data(dataset_name)
    _validate_length(raw_data, n_train_days, n_val_days, n_test_days, day_slot)

    train_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=0,
        num_days=n_train_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )
    val_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=n_train_days,
        num_days=n_val_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )
    test_sequences = generate_daily_sequences(
        data=raw_data,
        start_day=n_train_days + n_val_days,
        num_days=n_test_days,
        n_his=n_his,
        n_pred=max_pred_steps,
        day_slot=day_slot,
    )

    scaler = _fit_global_scaler(train_sequences)
    train_sequences, val_sequences, test_sequences = _normalize_splits(
        scaler, train_sequences, val_sequences, test_sequences
    )
    x_train, y_train = split_one_step_training_data(train_sequences, n_his)

    # Validation and test sets keep all future targets for recursive metrics.
    val_sequences = torch.from_numpy(
        np.ascontiguousarray(val_sequences, dtype=np.float32)
    ).to(device)
    test_sequences = torch.from_numpy(
        np.ascontiguousarray(test_sequences, dtype=np.float32)
    ).to(device)

    return (
        scaler,
        torch.from_numpy(x_train).to(device),
        torch.from_numpy(y_train).to(device),
        val_sequences,
        test_sequences,
    )
