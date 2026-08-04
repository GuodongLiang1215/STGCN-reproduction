# Experiments

The experiment code is grouped by purpose:

- `verification/` audits the official graph, GSO construction, and checkpoint
  selection protocol used in the final reproduction.
- `ablation/` compares controlled graph-pipeline configurations.
- `analysis/` produces the tracked result tables and figures.
- `exploratory/` retains early one-off diagnostics for provenance. These scripts
  are not required to reproduce the headline results.

Run maintained scripts as modules from the repository root, for example:

```powershell
python -m experiments.verification.compare_gso_construction
python -m experiments.ablation.compare_graph_ablation
python -m experiments.analysis.summarize_multiseed_results
```
