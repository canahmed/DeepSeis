# DeepSeis Audit

Generated: `2026-05-23T23:57:18+00:00`

## Dataset Files
- `raw_mseed_days`: 181
- `processed_window_arrays`: 543
- `processed_stft_arrays`: 543
- `checkpoints`: 3
- `figures`: 10

## Catalog
- `events`: 542
- `start_time`: 2023-02-06 01:17:34.342000+00:00
- `end_time`: 2023-08-03 04:56:18.057000+00:00
- `mw_ge_3_5`: 542
- `mw_ge_4_0`: 533
- `mw_ge_4_5`: 179
- `max_magnitude`: 7.8

## Preprocessing
- `reported_total_days`: 178
- `reported_errors`: 0
- `reported_total_windows`: 1522897
- `reported_total_spectrograms`: 1522897

## Labels
| Label | Total | Positive | Negative | Positive % |
|---|---:|---:|---:|---:|
| `labels_12h` | 162337 | 63202 | 99135 | 38.933 |
| `labels_24h` | 162337 | 89385 | 72952 | 55.061 |
| `labels_3h` | 162337 | 26281 | 136056 | 16.189 |
| `labels_6h` | 162337 | 41591 | 120746 | 25.620 |

## Labeling Matrix Candidates
| Min Mw | Hours | Matched Events | Positive % |
|---:|---:|---:|---:|
| 3.5 | 1 | 200 | 7.287 |
| 4.0 | 1 | 197 | 7.139 |
| 4.0 | 3 | 227 | 15.815 |
| 3.5 | 3 | 233 | 16.189 |

## Splits
| Split | Shape | Samples | Positive | Negative | Positive % | Size MB |
|---|---|---:|---:|---:|---:|---:|
| `train` | `[123422, 129, 95]` | 123422 | 38294 | 85128 | 31.027 | 5769.87 |
| `val` | `[17077, 129, 95]` | 17077 | 1794 | 15283 | 10.505 | 798.33 |
| `test` | `[21838, 129, 95]` | 21838 | 1503 | 20335 | 6.882 | 1020.91 |

## Model Metrics
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| `LSTM` | 0.070 | 0.049 | 0.058 | 0.423 | 0.060 |
| `TRANSFORMER` | 0.080 | 0.092 | 0.086 | 0.492 | 0.078 |

## Baseline Metrics
| Baseline | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| `always_negative` | 0.000 | 0.000 | 0.000 | 0.500 | 0.071 |
| `train_prior_constant` | 0.000 | 0.000 | 0.000 | 0.500 | 0.071 |
| `train_prior_random` | 0.069 | 0.313 | 0.113 | 0.494 | 0.068 |
| `stft_mean_energy` | 0.058 | 0.253 | 0.095 | 0.487 | 0.062 |

## Label Experiments
| Experiment | Min Mw | Hours | Total +% | Train +% | Val +% | Test +% |
|---|---:|---:|---:|---:|---:|---:|
| `mw40_1h_hhz` | 4.0 | 1 | 7.139 | 9.051 | 1.048 | 1.099 |
| `mw45_6h_hhz` | 4.5 | 6 | 9.808 | 12.318 | 0.0 | 3.292 |

## Thesis Placeholders
- Total placeholders: 11
- `tez/chapters/bulgular.tex`: 4
- `tez/chapters/giris.tex`: 1
- `tez/chapters/literatur.tex`: 2
- `tez/chapters/yontem.tex`: 4

## Warnings
- Train/test positive-label distribution drift is high (31.03% vs 6.88%).
- LSTM ROC-AUC is at or below random baseline (0.423).
- LSTM F1 score is very low (0.058).
- TRANSFORMER ROC-AUC is at or below random baseline (0.492).
- TRANSFORMER F1 score is very low (0.086).
- LSTM F1 (0.058) is below best simple baseline F1 (0.113).
- TRANSFORMER F1 (0.086) is below best simple baseline F1 (0.113).
- 24h labeling marks more than half of windows as positive; it is weak for anomaly discrimination.
- Preprocessing reports many more windows than labels; current labeling/splits appear to use a subset (likely HHZ only).
- Three-component processed arrays exist, but current split/label workflow should be checked for channel usage.
- Thesis still has 11 figure/table placeholders.
