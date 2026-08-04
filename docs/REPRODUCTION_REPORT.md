# STGCN Reproduction Report

## Scope

This project reproduces the STGCN(Cheb) traffic-forecasting results reported by
Yu, Yin, and Zhu on PeMSD7(M). The work began with the PyTorch implementation in
[hazdzz/STGCN](https://github.com/hazdzz/STGCN) and audited it against the
authors' [TensorFlow repository](https://github.com/VeritasYin/STGCN_IJCAI-18).

The audit covers four questions:

1. Does the data pipeline match the original experiment?
2. Does the PyTorch model preserve the TensorFlow layer semantics?
3. Which graph-construction differences affect the reported metrics?
4. Does checkpoint selection follow the original validation logic?

The objective is a controlled cross-framework reproduction, not bitwise
identity between TensorFlow 1 and current PyTorch.

## Dataset and forecasting protocol

PeMSD7(M) contains 44 days of traffic-speed measurements from 228 sensors at
five-minute intervals. The audited pipeline uses:

- `header=None` when loading `vel.csv`, preserving all 12,672 observations;
- the original 34/5/5-day train/validation/test split;
- daily sliding windows, so no sample crosses midnight;
- one mean and one standard deviation computed from the training sequences;
- 12 historical observations for one-step training; and
- nine recursive prediction steps at evaluation time.

Predictions at recursive steps 3, 6, and 9 correspond to the 15-, 30-, and
45-minute horizons. Each predicted value is appended to the history before the
next step, so later horizons include accumulated model error.

| Property | Value |
|---|---:|
| Sensors | 228 |
| Sampling interval | 5 minutes |
| Observations | 12,672 |
| Training sequences | 9,112 |
| Validation sequences | 1,340 |
| Test sequences | 1,340 |

## Model and optimization

The PyTorch model implements two ST-Conv blocks followed by the output layer:

```text
Input [B, 1, 12, 228]
  -> ST-Conv 1: 1 -> 32 -> 32 -> 64
  -> ST-Conv 2: 64 -> 32 -> 32 -> 128
  -> Output [B, 1, 1, 228]
```

Each ST-Conv block contains a temporal GLU, Chebyshev graph convolution,
temporal ReLU, and LayerNorm. The audited model has 333,604 trainable
parameters.

Training follows the original experiment where practical:

| Setting | Value |
|---|---:|
| Temporal kernel | 3 |
| Chebyshev order | 3 |
| Batch size | 50 |
| Epochs | 50 |
| Optimizer | RMSProp |
| Initial learning rate | 0.001 |
| Schedule | multiply by 0.7 every 5 epochs |
| Dropout | 0 |

The objective uses TensorFlow's `tf.nn.l2_loss` convention:

```text
0.5 * sum((prediction - target)^2)
```

This is a summed half-squared error rather than mean squared error.

## Graph-pipeline audit

The initial PyTorch graph and the graph distributed with the original STGCN
code are materially different. The third-party `adj.npz` contains 19,118
non-zero entries; the official graph contains 1,664. This difference motivated
a controlled comparison of adjacency source and graph shift operator (GSO)
construction.

| Experiment | Adjacency matrix | GSO |
|---|---|---|
| A | Third-party PyTorch graph | Current PyTorch utility |
| B | Official STGCN graph | Current PyTorch utility |
| C | Official STGCN graph | Original TensorFlow formulation |

The change from A to B isolates the adjacency-matrix effect. The change from B
to C isolates the GSO-construction effect.

Using the official graph reduced the 45-minute MAE by 0.134 and RMSE by 0.253
in the single-seed controlled experiment. Replacing the current GSO with the
original TensorFlow construction changed the corresponding metrics by only
0.006 and 0.001. The GSO implementations differ mainly at isolated nodes 13,
135, and 226, but that difference was small relative to the adjacency source.

Complete tables and plots are stored under
[`results/graph_ablation_comparison`](../results/graph_ablation_comparison),
with lower-level matrix checks in
[`results/graph_matrix_verification`](../results/graph_matrix_verification) and
[`results/gso_construction_verification`](../results/gso_construction_verification).

## Checkpoint-selection audit

The original TensorFlow code tracks a nine-element validation vector containing
MAPE, MAE, and RMSE for each horizon. Whenever any element improves, it records
the complete test vector from that epoch. Applying the same trigger rule to the
three runs selected:

| Seed | Selected epoch |
|---:|---:|
| 42 | 50 |
| 123 | 49 |
| 2026 | 48 |

Using the last epoch for every seed would therefore not match the original
tracking procedure. The trigger epochs and metric-level improvements are saved
inside each run's `selection_protocol` directory.

## Three-seed results

The final configuration combines the official PeMSD7(M) graph, the original
TensorFlow scaled-Laplacian construction, the audited PyTorch model, and the
original checkpoint-selection rule.

| Horizon | Paper MAPE | Reproduction MAPE | Paper MAE | Reproduction MAE | Paper RMSE | Reproduction RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 15 min | 5.260% | 5.237 ± 0.019% | 2.250 | 2.243 ± 0.014 | 4.040 | 4.064 ± 0.014 |
| 30 min | 7.330% | 7.392 ± 0.069% | 3.030 | 3.052 ± 0.048 | 5.700 | 5.743 ± 0.062 |
| 45 min | 8.690% | 8.896 ± 0.075% | 3.570 | 3.634 ± 0.078 | 6.770 | 6.856 ± 0.120 |

Values are mean and sample standard deviation over three seeds. Relative to the
paper, the mean MAPE differences are -0.023, +0.062, and +0.206 percentage
points at 15, 30, and 45 minutes. The standard deviation increases with horizon,
as expected for recursive prediction.

The detailed statistics are available in
[`results/multiseed_summary`](../results/multiseed_summary).

## Remaining differences

The remaining gap is small and concentrated at the 45-minute horizon. Plausible
sources include:

- TensorFlow 1 and modern PyTorch initialization and optimizer details;
- random-number generation and batch-shuffling order;
- CUDA kernel and floating-point accumulation order;
- sensitivity of MAPE at relatively low traffic speeds; and
- recursive error accumulation.

Three seeds provide an initial stability check but do not support strong claims
about statistical equivalence. The reproduction should be interpreted as a
close behavioral match across frameworks.

## Reproducing the analysis

Run training from the repository root after installing a suitable PyTorch build
and the packages in `requirements-no-torch.txt`:

```powershell
python paper_main.py --graph_source official --gso_source original --epochs 50 --checkpoint_every 1 --seed 123 --run_name official_original_gso_seed123_protocol
```

Reconstruct the selection protocol, evaluate its selected checkpoint, and
regenerate the summary:

```powershell
python -m experiments.verification.analyze_selection_protocol --run_name official_original_gso_seed123_protocol
python -m experiments.verification.evaluate_selected_checkpoint --run_name official_original_gso_seed123_protocol --epoch 49
python -m experiments.analysis.summarize_multiseed_results
```

Verification and ablation scripts are grouped under `experiments/`. Generated
checkpoints and smoke-test outputs are excluded from version control; tracked
CSV files, configurations, summaries, and selected figures provide the audit
trail for the reported results.

## Attribution

- Bing Yu, Haoteng Yin, and Zhanxing Zhu. *Spatio-Temporal Graph Convolutional
  Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018.
- Original implementation:
  [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18)
- Initial PyTorch baseline:
  [hazdzz/STGCN](https://github.com/hazdzz/STGCN)
