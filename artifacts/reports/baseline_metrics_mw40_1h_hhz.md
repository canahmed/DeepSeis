# DeepSeis Baseline Evaluation

Generated: `2026-05-23T23:56:36+00:00`

| Baseline | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---|---:|---:|---:|---:|---:|---|
| `always_negative` | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0118 | All windows are predicted as normal. |
| `train_prior_constant` | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0118 | Every window receives the training-set positive prior as score. |
| `train_prior_random` | 0.0120 | 0.1000 | 0.0214 | 0.4994 | 0.0109 | Random scores and random labels sampled using the training positive prior. |
| `stft_mean_energy` | 0.0058 | 0.0625 | 0.0105 | 0.5751 | 0.0120 | Mean log-STFT energy per window; threshold selected on validation F1. Threshold=0.6123. |
