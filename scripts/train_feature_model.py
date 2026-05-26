"""
Train fast feature-based classifiers for DeepSeis experiments.

This script extracts compact, deterministic features from STFT windows and
trains several scikit-learn classifiers. It is intended as a fast optimization
loop before committing to long deep-learning runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FEATURE_DIR = ARTIFACTS / "features"
REPORT_DIR = ARTIFACTS / "reports"
MODEL_DIR = ARTIFACTS / "models"


def resolve_x_path(splits_dir: Path, split: str) -> Path:
    direct = splits_dir / f"X_{split}.npy"
    if direct.exists():
        return direct

    manifest_path = splits_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ROOT / manifest["splits"][split]["x_path"]

    raise FileNotFoundError(f"Cannot resolve X_{split}.npy for {splits_dir}")


def experiment_name(splits_dir: Path) -> str:
    manifest_path = splits_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("name", splits_dir.name)
    return splits_dir.name


def window_features(batch: np.ndarray) -> np.ndarray:
    """Extract stable summary features from a batch shaped (N, F, T)."""
    x = np.asarray(batch, dtype=np.float32)
    n = x.shape[0]

    flat = x.reshape(n, -1)
    global_stats = np.stack([
        flat.mean(axis=1),
        flat.std(axis=1),
        flat.min(axis=1),
        flat.max(axis=1),
        np.percentile(flat, 10, axis=1),
        np.percentile(flat, 25, axis=1),
        np.percentile(flat, 50, axis=1),
        np.percentile(flat, 75, axis=1),
        np.percentile(flat, 90, axis=1),
    ], axis=1)

    freq_profile = x.mean(axis=2)
    time_profile = x.mean(axis=1)
    profile_stats = np.stack([
        freq_profile.mean(axis=1),
        freq_profile.std(axis=1),
        freq_profile.max(axis=1),
        time_profile.mean(axis=1),
        time_profile.std(axis=1),
        time_profile.max(axis=1),
    ], axis=1)

    freq_bands = np.array_split(np.arange(x.shape[1]), 8)
    time_bands = np.array_split(np.arange(x.shape[2]), 8)
    freq_band_means = np.stack([x[:, band, :].mean(axis=(1, 2)) for band in freq_bands], axis=1)
    time_band_means = np.stack([x[:, :, band].mean(axis=(1, 2)) for band in time_bands], axis=1)

    freq_grad = np.diff(freq_profile, axis=1)
    time_grad = np.diff(time_profile, axis=1)
    gradient_stats = np.stack([
        np.abs(freq_grad).mean(axis=1),
        np.abs(freq_grad).max(axis=1),
        np.abs(time_grad).mean(axis=1),
        np.abs(time_grad).max(axis=1),
    ], axis=1)

    grid = []
    for f_band in np.array_split(np.arange(x.shape[1]), 4):
        for t_band in np.array_split(np.arange(x.shape[2]), 4):
            grid.append(x[:, f_band][:, :, t_band].mean(axis=(1, 2)))
    grid_means = np.stack(grid, axis=1)

    features = np.concatenate([
        global_stats,
        profile_stats,
        freq_band_means,
        time_band_means,
        gradient_stats,
        grid_means,
    ], axis=1)
    return np.nan_to_num(features, copy=False).astype(np.float32)


def load_or_extract_features(splits_dir: Path, split: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    exp_name = experiment_name(splits_dir)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    x_cache = FEATURE_DIR / f"{exp_name}_{split}_stft_summary.npy"
    y_path = splits_dir / f"y_{split}.npy"
    y = np.load(y_path)

    if x_cache.exists():
        return np.load(x_cache, mmap_mode="r"), y

    x_path = resolve_x_path(splits_dir, split)
    x = np.load(x_path, mmap_mode="r")
    chunks = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunks.append(window_features(x[start:end]))
        print(f"[features] {split}: {end:,}/{x.shape[0]:,}", flush=True)

    features = np.concatenate(chunks, axis=0)
    np.save(x_cache, features)
    return features, y


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predicted_positive_rate": float(y_pred.mean()),
        "support": {
            "total": int(len(y_true)),
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
        },
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, max_alert_rate: float | None) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0, 1, 1001)))
    best_t = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred_rate = float((y_prob >= threshold).mean())
        if max_alert_rate is not None and pred_rate > max_alert_rate:
            continue
        score = f1_score(y_true, y_prob >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(threshold)
    if best_f1 < 0:
        best_f1 = float(f1_score(y_true, y_prob >= best_t, zero_division=0))
    return best_t, best_f1


def predict_prob(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def build_models(random_state: int) -> dict[str, object]:
    return {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "hist_gradient_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "extra_trees_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "random_forest_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train feature-based DeepSeis classifiers")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-alert-rate", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    exp_name = experiment_name(splits_dir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_or_extract_features(splits_dir, "train", args.chunk_size)
    x_val, y_val = load_or_extract_features(splits_dir, "val", args.chunk_size)
    x_test, y_test = load_or_extract_features(splits_dir, "test", args.chunk_size)

    results = {
        "experiment": exp_name,
        "splits_dir": str(splits_dir),
        "feature_count": int(x_train.shape[1]),
        "class_balance": {
            "train_positive_pct": float(y_train.mean() * 100),
            "val_positive_pct": float(y_val.mean() * 100),
            "test_positive_pct": float(y_test.mean() * 100),
        },
        "models": {},
    }

    best_model_name = None
    best_score = -1.0
    for name, model in build_models(args.random_state).items():
        print(f"[train] {name}", flush=True)
        model.fit(x_train, y_train)
        val_prob = predict_prob(model, x_val)
        test_prob = predict_prob(model, x_test)

        unconstrained_t, unconstrained_val_f1 = best_threshold(y_val, val_prob, max_alert_rate=None)
        constrained_t, constrained_val_f1 = best_threshold(y_val, val_prob, max_alert_rate=args.max_alert_rate)

        model_results = {
            "default_0_5": metrics_at_threshold(y_test, test_prob, 0.5),
            "val_f1": metrics_at_threshold(y_test, test_prob, unconstrained_t),
            "val_f1_constrained": metrics_at_threshold(y_test, test_prob, constrained_t),
            "thresholds": {
                "val_f1": {"threshold": unconstrained_t, "validation_f1": unconstrained_val_f1},
                "val_f1_constrained": {
                    "threshold": constrained_t,
                    "validation_f1": constrained_val_f1,
                    "max_alert_rate": args.max_alert_rate,
                },
            },
        }
        results["models"][name] = model_results

        score = model_results["val_f1_constrained"]["f1_score"]
        if score > best_score:
            best_score = score
            best_model_name = name

        joblib.dump(model, MODEL_DIR / f"{exp_name}_{name}.joblib")

    results["best_by_constrained_test_f1"] = best_model_name
    output_path = REPORT_DIR / f"feature_models_{exp_name}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
