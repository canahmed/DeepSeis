# DeepSeis Baseline Evaluation

Generated: `2026-05-23T23:42:14+00:00`

| Baseline | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---|---:|---:|---:|---:|---:|---|
| `always_negative` | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0714 | All windows are predicted as normal. |
| `train_prior_constant` | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0714 | Every window receives the training-set positive prior as score. |
| `train_prior_random` | 0.0687 | 0.3127 | 0.1126 | 0.4940 | 0.0677 | Random scores and random labels sampled using the training positive prior. |
| `stft_mean_energy` | 0.0584 | 0.2535 | 0.0950 | 0.4869 | 0.0622 | Mean log-STFT energy per window; threshold selected on validation F1. Threshold=0.1790. |
