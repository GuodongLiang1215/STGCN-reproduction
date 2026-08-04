# STGCN Reproducibility Gap Note

## Scope

This note summarizes the remaining differences between a paper-aligned PyTorch reproduction of STGCN on PeMSD7(M) and the results reported for STGCN(Cheb).

The reproduction aligns the following components:

- PeMSD7(M) data loading with `header=None`;
- 34/5/5-day train-validation-test split;
- daily sequence generation without crossing midnight;
- global training-set Z-score normalization;
- one-step training with 9-step autoregressive evaluation;
- the paper-aligned STGCN architecture;
- RMSProp optimization and learning-rate decay;
- the official PeMSD7(M) graph construction;
- the original TensorFlow scaled-Laplacian construction;
- the original validation-trigger model-selection logic.

## Three-Seed Results

The final statistics use the original validation-trigger protocol with the following selected checkpoints:

- seed 42: epoch 50;
- seed 123: epoch 49;
- seed 2026: epoch 48.

| Horizon | MAPE (%) | MAE | RMSE |
|---|---:|---:|---:|
| 15 min | 5.237 ± 0.019 | 2.243 ± 0.014 | 4.064 ± 0.014 |
| 30 min | 7.392 ± 0.069 | 3.052 ± 0.048 | 5.743 ± 0.062 |
| 45 min | 8.896 ± 0.075 | 3.634 ± 0.078 | 6.856 ± 0.120 |

Values are mean ± sample standard deviation over three random seeds.

## Difference from the Paper

| Horizon | Metric | Paper | Reproduction Mean | Difference |
|---|---|---:|---:|---:|
| 15 min | MAPE | 5.260% | 5.237% | -0.023 pp |
| 15 min | MAE | 2.250 | 2.243 | -0.007 |
| 15 min | RMSE | 4.040 | 4.064 | +0.024 |
| 30 min | MAPE | 7.330% | 7.392% | +0.062 pp |
| 30 min | MAE | 3.030 | 3.052 | +0.022 |
| 30 min | RMSE | 5.700 | 5.743 | +0.043 |
| 45 min | MAPE | 8.690% | 8.896% | +0.206 pp |
| 45 min | MAE | 3.570 | 3.634 | +0.064 |
| 45 min | RMSE | 6.770 | 6.856 | +0.086 |

The largest remaining discrepancy is the 45-minute MAPE.

## Findings from Controlled Experiments

### 1. Adjacency-matrix mismatch was the main implementation gap

The initial PyTorch adjacency matrix contained 19,118 non-zero values, while the official STGCN graph contained 1,664 non-zero values.

Replacing the initial graph with the official graph improved all MAE and RMSE values, with the largest improvement at the 45-minute horizon.

This indicates that graph construction was a major source of the initial reproduction gap.

### 2. GSO construction had a much smaller effect

The current PyTorch graph operator and the original TensorFlow scaled Laplacian differ mainly at three isolated nodes: 13, 135, and 226.

Switching from the current PyTorch GSO to the original TensorFlow GSO changed MAE and RMSE only slightly. Therefore, the adjacency-matrix source had a larger practical effect than the isolated-node GSO difference.

### 3. Model-selection protocol matters

The original repository updates the complete test metric vector whenever any validation metric improves.

Under this protocol:

- seed 42 selected epoch 50;
- seed 123 selected epoch 49;
- seed 2026 selected epoch 48.

Using the final epoch for every seed would therefore not reproduce the original tracking behavior.

### 4. Random-seed variation increases with forecast horizon

The standard deviations are small at 15 minutes and larger at 30 and 45 minutes.

This is consistent with autoregressive error accumulation: later forecasts depend on more model-generated inputs, so small differences in learned parameters can propagate through repeated prediction steps.

## Remaining Plausible Sources of Difference

The remaining small gap may be related to:

- TensorFlow 1 versus modern PyTorch numerical behavior;
- framework-specific parameter initialization;
- RMSProp implementation details;
- random-number generation and batch-shuffling order;
- CUDA kernel and floating-point accumulation order;
- MAPE sensitivity when true traffic speeds are relatively low;
- recursive error accumulation at the 45-minute horizon.

## Current Conclusion

The PyTorch reproduction closely matches the reported STGCN(Cheb) results across three random seeds.

The controlled experiments show that the official adjacency matrix was the most important correction. The GSO implementation difference had a comparatively small effect. After graph-pipeline alignment and original-protocol checkpoint selection, the remaining discrepancy is small and concentrated mainly in the 45-minute MAPE.

The reproduction should therefore be described as a successful paper-aligned cross-framework reproduction, rather than a bitwise-identical reimplementation.
