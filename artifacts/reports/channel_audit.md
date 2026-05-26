# DeepSeis Channel Audit

Generated: `2026-05-23T23:36:51+00:00`

## Summary
- Dates with any processed metadata: 181
- Dates with all three components: 181
- Dates with equal HHZ/HHN/HHE counts: 13
- Dates with mismatched component counts: 168
- Dates with missing components: 0

## Channel Totals
| Channel | Metadata windows |
|---|---:|
| `HHZ` | 162337 |
| `HHN` | 123653 |
| `HHE` | 113475 |

## Label Totals
| Label file | Windows |
|---|---:|
| `labels_12h` | 162337 |
| `labels_24h` | 162337 |
| `labels_3h` | 162337 |
| `labels_6h` | 162337 |

## Sample Mismatched Dates
| Date | HHZ | HHN | HHE |
|---|---:|---:|---:|
| `2023-02-06` | 1321 | 575 | 23 |
| `2023-02-08` | 1917 | 96 | 705 |
| `2023-02-09` | 1871 | 1042 | 416 |
| `2023-02-10` | 554 | 54 | 15 |
| `2023-02-11` | 183 | 94 | 684 |
| `2023-02-12` | 506 | 1520 | 506 |
| `2023-02-13` | 1584 | 2879 | 2879 |
| `2023-02-15` | 377 | 377 | 376 |
| `2023-02-16` | 522 | 522 | 469 |
| `2023-02-17` | 361 | 156 | 497 |
| `2023-02-18` | 1395 | 1255 | 511 |
| `2023-02-19` | 907 | 1433 | 347 |
| `2023-02-20` | 1244 | 1227 | 650 |
| `2023-02-21` | 1475 | 357 | 1465 |
| `2023-02-22` | 372 | 1615 | 373 |
| `2023-02-23` | 825 | 471 | 471 |
| `2023-02-24` | 1468 | 1688 | 1688 |
| `2023-02-25` | 247 | 235 | 247 |
| `2023-02-27` | 2879 | 1376 | 692 |
| `2023-03-01` | 404 | 404 | 413 |

## Recommendation
- Current labels align exactly with HHZ metadata counts.
- Naively stacking HHZ/HHN/HHE by array index is unsafe because many dates have different valid-window counts.
- For a three-component model, rebuild preprocessing around common time windows or align by metadata timestamps.
- Near-term model improvements should either document HHZ-only training or create a new timestamp-aligned 3C dataset.
