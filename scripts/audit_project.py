"""
DeepSeis project audit.

Creates a lightweight JSON and Markdown snapshot of the current dataset,
labels, splits, trained metrics, and thesis placeholders. The script is
read-only with respect to project data; it only writes reports under
artifacts/reports.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "reports"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_count(pattern: str, base: Path) -> int:
    return sum(1 for _ in base.glob(pattern)) if base.exists() else 0


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def npy_shape(path: Path) -> list[int]:
    arr = np.load(path, mmap_mode="r")
    return list(arr.shape)


def label_stats(path: Path) -> dict[str, Any]:
    arr = np.load(path, mmap_mode="r")
    total = int(arr.shape[0])
    positive = int(arr.sum())
    negative = total - positive
    return {
        "file": rel(path),
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_pct": round(positive / total * 100, 3) if total else 0.0,
    }


def split_stats(splits_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        x_path = splits_dir / f"X_{split}.npy"
        y_path = splits_dir / f"y_{split}.npy"
        if not x_path.exists() or not y_path.exists():
            continue
        y = np.load(y_path, mmap_mode="r")
        total = int(y.shape[0])
        positive = int(y.sum())
        result[split] = {
            "x_file": rel(x_path),
            "y_file": rel(y_path),
            "x_shape": npy_shape(x_path),
            "samples": total,
            "positive": positive,
            "negative": total - positive,
            "positive_pct": round(positive / total * 100, 3) if total else 0.0,
            "x_size_mb": round(x_path.stat().st_size / 1024 / 1024, 2),
        }
    return result


def catalog_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    magnitudes = [float(row["magnitude"]) for row in rows if row.get("magnitude")]
    times = sorted(row["time"] for row in rows if row.get("time"))
    return {
        "file": rel(path),
        "events": len(rows),
        "start_time": times[0] if times else None,
        "end_time": times[-1] if times else None,
        "mw_ge_3_5": sum(1 for mag in magnitudes if mag >= 3.5),
        "mw_ge_4_0": sum(1 for mag in magnitudes if mag >= 4.0),
        "mw_ge_4_5": sum(1 for mag in magnitudes if mag >= 4.5),
        "max_magnitude": max(magnitudes) if magnitudes else None,
    }


def metrics_stats(metrics_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        if path.name == "baseline_metrics.json":
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        model = path.stem.replace("_metrics", "")
        result[model] = {
            "file": rel(path),
            "accuracy": data.get("accuracy"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "f1_score": data.get("f1_score"),
            "roc_auc": data.get("roc_auc"),
            "pr_auc": data.get("pr_auc"),
            "support": data.get("support"),
            "confusion_matrix": data.get("confusion_matrix"),
        }
    return result


def baseline_stats(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    baselines = data.get("baselines", {})
    if not isinstance(baselines, dict):
        return {}
    result: dict[str, Any] = {}
    for name, metrics in baselines.items():
        if not isinstance(metrics, dict):
            continue
        result[name] = {
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
        }
    return result


def labeling_matrix_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        "channel": data.get("channel"),
        "window_count": data.get("window_count"),
        "recommended_candidates": data.get("recommended_candidates", []),
    }


def experiment_summaries(experiments_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not experiments_dir.exists():
        return result
    for manifest_path in sorted(experiments_dir.glob("*/manifest.json")):
        data = load_json(manifest_path)
        if not isinstance(data, dict):
            continue
        name = data.get("name") or manifest_path.parent.name
        result[name] = {
            "path": rel(manifest_path.parent),
            "channel": data.get("channel"),
            "min_magnitude": data.get("min_magnitude"),
            "pre_event_hours": data.get("pre_event_hours"),
            "total_windows": data.get("total_windows"),
            "positive": data.get("positive"),
            "positive_pct": data.get("positive_pct"),
            "splits": {
                split: {
                    "samples": info.get("samples"),
                    "positive": info.get("positive"),
                    "positive_pct": info.get("positive_pct"),
                }
                for split, info in data.get("splits", {}).items()
            },
        }
    return result


def preprocessing_stats(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        "file": rel(path),
        "reported_total_days": data.get("total_days"),
        "reported_errors": data.get("errors"),
        "reported_total_windows": data.get("total_windows"),
        "reported_total_spectrograms": data.get("total_spectrograms"),
    }


def thesis_placeholder_stats(thesis_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"placeholder_count": 0, "files": {}}
    if not thesis_dir.exists():
        return result
    pattern = re.compile(r"\[.*?EKLENECEK.*?\]", re.IGNORECASE)
    for path in sorted(thesis_dir.rglob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = pattern.findall(text)
        if matches:
            result["files"][rel(path)] = len(matches)
            result["placeholder_count"] += len(matches)
    return result


def build_warnings(audit: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    splits = audit.get("splits", {})
    train = splits.get("train", {})
    test = splits.get("test", {})
    if train and test:
        drift = abs(float(train["positive_pct"]) - float(test["positive_pct"]))
        if drift >= 10.0:
            warnings.append(
                "Train/test positive-label distribution drift is high "
                f"({train['positive_pct']:.2f}% vs {test['positive_pct']:.2f}%)."
            )

    for model, metrics in audit.get("model_metrics", {}).items():
        roc_auc = metrics.get("roc_auc")
        f1 = metrics.get("f1_score")
        if roc_auc is not None and float(roc_auc) <= 0.5:
            warnings.append(f"{model} ROC-AUC is at or below random baseline ({float(roc_auc):.3f}).")
        if f1 is not None and float(f1) < 0.10:
            warnings.append(f"{model} F1 score is very low ({float(f1):.3f}).")

    baselines = audit.get("baseline_metrics", {})
    if baselines:
        best_baseline_f1 = max(float(item.get("f1_score") or 0.0) for item in baselines.values())
        for model, metrics in audit.get("model_metrics", {}).items():
            model_f1 = float(metrics.get("f1_score") or 0.0)
            if model_f1 < best_baseline_f1:
                warnings.append(
                    f"{model} F1 ({model_f1:.3f}) is below best simple baseline F1 ({best_baseline_f1:.3f})."
                )

    labels = audit.get("labels", {})
    labels_24h = labels.get("labels_24h", {})
    if labels_24h and float(labels_24h.get("positive_pct", 0.0)) > 50.0:
        warnings.append("24h labeling marks more than half of windows as positive; it is weak for anomaly discrimination.")

    file_counts = audit.get("files", {})
    preprocessing = audit.get("preprocessing", {})
    reported_windows = preprocessing.get("reported_total_windows")
    if reported_windows and labels:
        largest_label_total = max(int(item["total"]) for item in labels.values())
        if int(reported_windows) > largest_label_total * 3:
            warnings.append(
                "Preprocessing reports many more windows than labels; current labeling/splits appear to use a subset "
                "(likely HHZ only)."
            )
    if file_counts.get("processed_window_arrays", 0) >= file_counts.get("raw_mseed_days", 0) * 3 and labels:
        warnings.append("Three-component processed arrays exist, but current split/label workflow should be checked for channel usage.")

    thesis = audit.get("thesis", {})
    if thesis.get("placeholder_count", 0):
        warnings.append(f"Thesis still has {thesis['placeholder_count']} figure/table placeholders.")

    return warnings


def build_audit() -> dict[str, Any]:
    windows_dir = ROOT / "data" / "processed" / "windows"
    stft_dir = ROOT / "data" / "processed" / "stft"
    labels = {
        path.stem: label_stats(path)
        for path in sorted(windows_dir.glob("labels_*h.npy"))
    } if windows_dir.exists() else {}

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paths": {
            "root": str(ROOT),
            "reports": rel(REPORTS_DIR),
        },
        "files": {
            "raw_mseed_days": file_count("*.mseed", ROOT / "data" / "raw" / "mseed"),
            "processed_window_arrays": file_count("*_windows.npy", windows_dir),
            "processed_stft_arrays": file_count("*_stft.npy", stft_dir),
            "checkpoints": file_count("*.pt", ROOT / "artifacts" / "checkpoints"),
            "figures": file_count("*.png", ROOT / "artifacts" / "figures"),
        },
        "preprocessing": preprocessing_stats(REPORTS_DIR / "preprocessing_stats.json"),
        "catalog": catalog_stats(ROOT / "data" / "raw" / "catalog" / "earthquake_catalog.csv"),
        "labels": labels,
        "split_report": load_json(REPORTS_DIR / "split_report.json"),
        "splits": split_stats(windows_dir / "splits"),
        "model_metrics": metrics_stats(REPORTS_DIR),
        "baseline_metrics": baseline_stats(REPORTS_DIR / "baseline_metrics.json"),
        "labeling_matrix": labeling_matrix_summary(REPORTS_DIR / "labeling_matrix.json"),
        "experiments": experiment_summaries(ROOT / "data" / "processed" / "experiments"),
        "thesis": thesis_placeholder_stats(ROOT / "tez"),
    }
    audit["warnings"] = build_warnings(audit)
    return audit


def fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# DeepSeis Audit")
    lines.append("")
    lines.append(f"Generated: `{audit['generated_at_utc']}`")
    lines.append("")
    lines.append("## Dataset Files")
    for key, value in audit["files"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Catalog")
    catalog = audit.get("catalog", {})
    for key in ("events", "start_time", "end_time", "mw_ge_3_5", "mw_ge_4_0", "mw_ge_4_5", "max_magnitude"):
        lines.append(f"- `{key}`: {catalog.get(key)}")
    lines.append("")
    lines.append("## Preprocessing")
    preprocessing = audit.get("preprocessing", {})
    for key in ("reported_total_days", "reported_errors", "reported_total_windows", "reported_total_spectrograms"):
        lines.append(f"- `{key}`: {preprocessing.get(key)}")
    lines.append("")
    lines.append("## Labels")
    lines.append("| Label | Total | Positive | Negative | Positive % |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, stats in audit.get("labels", {}).items():
        lines.append(
            f"| `{name}` | {stats['total']} | {stats['positive']} | "
            f"{stats['negative']} | {stats['positive_pct']:.3f} |"
        )
    lines.append("")
    lines.append("## Labeling Matrix Candidates")
    lines.append("| Min Mw | Hours | Matched Events | Positive % |")
    lines.append("|---:|---:|---:|---:|")
    for row in audit.get("labeling_matrix", {}).get("recommended_candidates", []):
        lines.append(
            f"| {row['min_magnitude']:.1f} | {row['hours']} | "
            f"{row['matched_events']} | {row['positive_pct']:.3f} |"
        )
    lines.append("")
    lines.append("## Splits")
    lines.append("| Split | Shape | Samples | Positive | Negative | Positive % | Size MB |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, stats in audit.get("splits", {}).items():
        lines.append(
            f"| `{name}` | `{stats['x_shape']}` | {stats['samples']} | "
            f"{stats['positive']} | {stats['negative']} | "
            f"{stats['positive_pct']:.3f} | {stats['x_size_mb']:.2f} |"
        )
    lines.append("")
    lines.append("## Model Metrics")
    lines.append("| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, stats in audit.get("model_metrics", {}).items():
        lines.append(
            f"| `{name}` | {fmt_pct(stats.get('precision'))} | "
            f"{fmt_pct(stats.get('recall'))} | {fmt_pct(stats.get('f1_score'))} | "
            f"{fmt_pct(stats.get('roc_auc'))} | {fmt_pct(stats.get('pr_auc'))} |"
        )
    lines.append("")
    lines.append("## Baseline Metrics")
    lines.append("| Baseline | Precision | Recall | F1 | ROC-AUC | PR-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, stats in audit.get("baseline_metrics", {}).items():
        lines.append(
            f"| `{name}` | {fmt_pct(stats.get('precision'))} | "
            f"{fmt_pct(stats.get('recall'))} | {fmt_pct(stats.get('f1_score'))} | "
            f"{fmt_pct(stats.get('roc_auc'))} | {fmt_pct(stats.get('pr_auc'))} |"
        )
    lines.append("")
    lines.append("## Label Experiments")
    lines.append("| Experiment | Min Mw | Hours | Total +% | Train +% | Val +% | Test +% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, experiment in audit.get("experiments", {}).items():
        splits = experiment.get("splits", {})
        lines.append(
            f"| `{name}` | {experiment.get('min_magnitude')} | {experiment.get('pre_event_hours')} | "
            f"{experiment.get('positive_pct')} | "
            f"{splits.get('train', {}).get('positive_pct')} | "
            f"{splits.get('val', {}).get('positive_pct')} | "
            f"{splits.get('test', {}).get('positive_pct')} |"
        )
    lines.append("")
    lines.append("## Thesis Placeholders")
    thesis = audit.get("thesis", {})
    lines.append(f"- Total placeholders: {thesis.get('placeholder_count', 0)}")
    for file_name, count in thesis.get("files", {}).items():
        lines.append(f"- `{file_name}`: {count}")
    lines.append("")
    lines.append("## Warnings")
    warnings = audit.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()

    json_path = REPORTS_DIR / "project_audit.json"
    md_path = REPORTS_DIR / "project_audit.md"

    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, md_path)

    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")


if __name__ == "__main__":
    main()
