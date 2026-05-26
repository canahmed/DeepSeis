"""
Train a clean normal-vs-event feature model.

Positive class:
    Windows within +/- event_window_minutes of an Mw >= min_magnitude event.

Negative class:
    Windows at least negative_gap_hours away from every Mw >= min_magnitude event.

The resulting balanced dataset is a controlled anomaly/normal classification
task. It should be reported separately from chronological forecasting tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "data" / "processed" / "windows"
CATALOG_PATH = ROOT / "data" / "raw" / "catalog" / "earthquake_catalog.csv"
FEATURE_DIR = ROOT / "artifacts" / "features"
REPORT_DIR = ROOT / "artifacts" / "reports"
MODEL_DIR = ROOT / "artifacts" / "models"


def parse_time(value: str) -> np.datetime64:
    normalized = value.replace("Z", "+00:00")
    if "+" in normalized:
        normalized = normalized.split("+", 1)[0]
    return np.datetime64(normalized).astype("datetime64[ns]")


def load_centers(channel: str) -> np.ndarray:
    metadata = []
    for path in sorted(WINDOWS_DIR.glob(f"*_{channel}_metadata.json")):
        metadata.extend(json.loads(path.read_text(encoding="utf-8")))

    centers = np.empty(len(metadata), dtype="datetime64[ns]")
    for idx, item in enumerate(metadata):
        start = parse_time(item["start_time"])
        end = parse_time(item["end_time"])
        centers[idx] = start + (end - start) / 2
    return centers


def load_events(min_magnitude: float) -> list[np.datetime64]:
    events = []
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if float(row["magnitude"]) >= min_magnitude:
                events.append(parse_time(row["time"]))
    return events


def load_features(experiment: str) -> np.ndarray:
    chunks = []
    for split in ["train", "val", "test"]:
        path = FEATURE_DIR / f"{experiment}_{split}_stft_summary.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {path}. "
                "Run scripts/train_feature_model.py for this experiment first."
            )
        chunks.append(np.load(path))
    return np.concatenate(chunks, axis=0)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0, 1, 1001)))
    best_t = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        score = f1_score(y_true, y_prob >= threshold, zero_division=0)
        if score > best_f1:
            best_t = float(threshold)
            best_f1 = float(score)
    return best_t, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support": {
            "total": int(len(y_true)),
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
        },
    }


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "hist_gradient": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=500,
                l2_regularization=0.02,
                random_state=random_state,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=700,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train clean balanced feature models")
    parser.add_argument("--experiment", default="mw40_pm60m_hhz")
    parser.add_argument("--channel", default="HHZ")
    parser.add_argument("--min-magnitude", type=float, default=4.0)
    parser.add_argument("--event-window-minutes", type=int, default=60)
    parser.add_argument("--negative-gap-hours", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=11)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    centers = load_centers(args.channel.upper())
    events = load_events(args.min_magnitude)
    positive = np.zeros(len(centers), dtype=bool)
    near_event = np.zeros(len(centers), dtype=bool)
    event_delta = np.timedelta64(args.event_window_minutes, "m")
    gap_delta = np.timedelta64(args.negative_gap_hours, "h")

    for event_time in events:
        positive |= (centers >= event_time - event_delta) & (centers <= event_time + event_delta)
        near_event |= (centers >= event_time - gap_delta) & (centers <= event_time + gap_delta)

    negative_pool = ~near_event
    positive_idx = np.where(positive)[0]
    negative_idx = np.where(negative_pool)[0]
    if len(negative_idx) < len(positive_idx):
        raise ValueError("Not enough clean negative windows for a balanced dataset.")

    rng = np.random.default_rng(args.random_state)
    sampled_negative_idx = rng.choice(negative_idx, size=len(positive_idx), replace=False)
    selected_idx = np.concatenate([positive_idx, sampled_negative_idx])
    rng.shuffle(selected_idx)

    labels = np.zeros(len(centers), dtype=np.int8)
    labels[positive] = 1
    features = load_features(args.experiment)

    train_idx, test_idx = train_test_split(
        selected_idx,
        test_size=0.20,
        random_state=args.random_state,
        stratify=labels[selected_idx],
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.20,
        random_state=args.random_state,
        stratify=labels[train_idx],
    )

    results = {
        "protocol": "clean_balanced_event_vs_far_normal",
        "experiment": args.experiment,
        "channel": args.channel.upper(),
        "min_magnitude": args.min_magnitude,
        "event_window_minutes": args.event_window_minutes,
        "negative_gap_hours": args.negative_gap_hours,
        "positive_count": int(len(positive_idx)),
        "negative_pool": int(len(negative_idx)),
        "sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "models": {},
    }

    best_name = None
    best_f1 = -1.0
    for name, model in build_models(args.random_state).items():
        print(f"[train] {name}", flush=True)
        model.fit(features[train_idx], labels[train_idx])
        val_prob = model.predict_proba(features[val_idx])[:, 1]
        threshold, val_f1 = best_threshold(labels[val_idx], val_prob)
        test_prob = model.predict_proba(features[test_idx])[:, 1]
        metrics = compute_metrics(labels[test_idx], test_prob, threshold)
        metrics["validation_f1_at_threshold"] = val_f1
        results["models"][name] = metrics

        model_path = MODEL_DIR / (
            f"{args.experiment}_clean_gap{args.negative_gap_hours}h_{name}.joblib"
        )
        joblib.dump({
            "model": model,
            "threshold": threshold,
            "protocol": results["protocol"],
            "experiment": args.experiment,
            "feature_set": "stft_summary_v1",
        }, model_path)

        if metrics["f1_score"] > best_f1:
            best_name = name
            best_f1 = metrics["f1_score"]

    results["best_by_test_f1"] = best_name
    report_path = REPORT_DIR / (
        f"clean_balanced_{args.experiment}_gap{args.negative_gap_hours}h_feature_models.json"
    )
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {report_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
