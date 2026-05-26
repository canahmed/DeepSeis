"""
Evaluate simple DeepSeis baselines.

These baselines are intentionally modest. They answer a crucial thesis question:
do trained models beat trivial class-prior and simple signal-energy references?
The script reads existing split arrays and writes reports under artifacts/reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = ROOT / "data" / "processed" / "windows" / "splits"
REPORTS_DIR = ROOT / "artifacts" / "reports"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int8)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp_cumsum = np.cumsum(sorted_true)
    precision = tp_cumsum / (np.arange(len(y_true)) + 1)
    return float((precision * sorted_true).sum() / n_pos)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc_score_binary(y_true, y_score),
        "pr_auc": average_precision_score_binary(y_true, y_score),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {
            "total": int(total),
            "positive": int(y_true.sum()),
            "negative": int(total - y_true.sum()),
        },
    }


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(y_score)
    if len(thresholds) > 500:
        thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, 501))
        thresholds = np.unique(thresholds)

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(np.int8)
        score = metrics(y_true, pred, y_score)["f1_score"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def load_manifest(splits_dir: Path) -> dict[str, Any] | None:
    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_x_path(splits_dir: Path, split: str, manifest: dict[str, Any] | None) -> Path:
    direct_path = splits_dir / f"X_{split}.npy"
    if direct_path.exists():
        return direct_path
    if manifest and split in manifest.get("splits", {}):
        return ROOT / manifest["splits"][split]["x_path"]
    return direct_path


def mean_stft_energy(splits_dir: Path, split: str, manifest: dict[str, Any] | None, chunk_size: int = 4096) -> np.ndarray:
    path = resolve_x_path(splits_dir, split, manifest)
    x = np.load(path, mmap_mode="r")
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = x[start:end].mean(axis=(1, 2))
    return out


def robust_zscore(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    median = float(np.median(train_scores))
    mad = float(np.median(np.abs(train_scores - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(train_scores) + 1e-8)
    return (scores - median) / scale


def evaluate(splits_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(splits_dir)
    y_train = np.load(splits_dir / "y_train.npy")
    y_val = np.load(splits_dir / "y_val.npy")
    y_test = np.load(splits_dir / "y_test.npy")

    positive_prior = float(y_train.mean())
    rng = np.random.default_rng(42)

    baselines: dict[str, Any] = {}

    zero_pred = np.zeros_like(y_test, dtype=np.int8)
    zero_score = np.zeros_like(y_test, dtype=np.float32)
    baselines["always_negative"] = metrics(y_test, zero_pred, zero_score)
    baselines["always_negative"]["description"] = "All windows are predicted as normal."

    prior_score = np.full_like(y_test, fill_value=positive_prior, dtype=np.float32)
    prior_pred = (prior_score >= 0.5).astype(np.int8)
    baselines["train_prior_constant"] = metrics(y_test, prior_pred, prior_score)
    baselines["train_prior_constant"]["description"] = "Every window receives the training-set positive prior as score."
    baselines["train_prior_constant"]["positive_prior"] = positive_prior

    random_score = rng.random(len(y_test), dtype=np.float32)
    random_pred = (random_score < positive_prior).astype(np.int8)
    baselines["train_prior_random"] = metrics(y_test, random_pred, random_score)
    baselines["train_prior_random"]["description"] = "Random scores and random labels sampled using the training positive prior."
    baselines["train_prior_random"]["seed"] = 42

    train_energy = mean_stft_energy(splits_dir, "train", manifest)
    val_energy = mean_stft_energy(splits_dir, "val", manifest)
    test_energy = mean_stft_energy(splits_dir, "test", manifest)

    val_energy_z = robust_zscore(train_energy, val_energy)
    test_energy_z = robust_zscore(train_energy, test_energy)
    threshold, val_f1 = best_threshold_by_f1(y_val, val_energy_z)
    energy_pred = (test_energy_z >= threshold).astype(np.int8)
    baselines["stft_mean_energy"] = metrics(y_test, energy_pred, test_energy_z)
    baselines["stft_mean_energy"]["description"] = (
        "Mean log-STFT energy per window; threshold selected on validation F1."
    )
    baselines["stft_mean_energy"]["threshold_from_val"] = threshold
    baselines["stft_mean_energy"]["val_f1_at_threshold"] = val_f1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits_dir": rel(splits_dir),
        "experiment": manifest.get("name") if manifest else "canonical",
        "baselines": baselines,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DeepSeis Baseline Evaluation",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "| Baseline | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in report["baselines"].items():
        notes = result.get("description", "")
        if "threshold_from_val" in result:
            notes += f" Threshold={result['threshold_from_val']:.4f}."
        lines.append(
            f"| `{name}` | {fmt(result['precision'])} | {fmt(result['recall'])} | "
            f"{fmt(result['f1_score'])} | {fmt(result['roc_auc'])} | "
            f"{fmt(result['pr_auc'])} | {notes} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeis baselines")
    parser.add_argument("--splits-dir", type=str, default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--output-prefix", type=str, default="baseline_metrics")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = ROOT / splits_dir

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = evaluate(splits_dir)
    json_path = REPORTS_DIR / f"{args.output_prefix}.json"
    md_path = REPORTS_DIR / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")


if __name__ == "__main__":
    main()
