import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from flask import Flask, jsonify, render_template

from src.models.cnn_baseline import build_cnn
from src.models.gru_model import build_gru
from src.models.lstm_model import build_lstm
from src.models.transformer_model import build_transformer
from src.models.transformer_v2 import build_transformer_v2


app = Flask(__name__)

model = None
device = None
X_test = None
y_test = None
MODEL_BACKEND = "pytorch"
CURRENT_INDEX = 0
MAX_INDEX = 0
APP_STATE = {
    "ready": False,
    "backend": None,
    "model_arch": None,
    "checkpoint": None,
    "model_path": None,
    "splits_dir": None,
    "experiment": "canonical",
    "threshold": 0.5,
    "feature_set": None,
    "error": None,
}


def build_model(model_arch: str):
    if model_arch == "lstm":
        return build_lstm(freq_bins=129, time_steps=95, num_layers=2)
    if model_arch == "gru":
        return build_gru(freq_bins=129, time_steps=95, num_layers=2)
    if model_arch == "cnn":
        return build_cnn(freq_bins=129, time_steps=95)
    if model_arch == "transformer":
        return build_transformer(freq_bins=129, time_steps=95, d_model=128, nhead=4, num_encoder_layers=2)
    if model_arch == "transformer_v2":
        return build_transformer_v2(
            freq_bins=129, time_steps=95,
            cnn_channels=64, d_model=256, nhead=8,
            num_encoder_layers=4, dim_feedforward=512,
            dropout=0.2
        )
    raise ValueError(f"Unknown model architecture: {model_arch}")


def load_manifest(splits_dir: Path):
    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_x_test_path(splits_dir: Path, manifest):
    direct_path = splits_dir / "X_test.npy"
    if direct_path.exists():
        return direct_path
    if manifest and "test" in manifest.get("splits", {}):
        return Path(manifest["splits"]["test"]["x_path"])
    return direct_path


def stft_summary_features(batch: np.ndarray) -> np.ndarray:
    """Extract the same 51 STFT summary features used by train_feature_model.py."""
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


def init_system():
    global model, device, X_test, y_test, MAX_INDEX, APP_STATE, MODEL_BACKEND

    try:
        with open("configs/station_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_arch = os.getenv("DEEPSEIS_MODEL", "transformer").lower()
        MODEL_BACKEND = os.getenv("DEEPSEIS_BACKEND", "pytorch").lower()
        splits_dir = Path(os.getenv(
            "DEEPSEIS_SPLITS_DIR",
            str(Path(config["paths"]["processed_windows"]) / "splits")
        ))
        checkpoint_path = Path(os.getenv(
            "DEEPSEIS_CHECKPOINT",
            f"artifacts/checkpoints/{model_arch}_best.pt"
        ))
        feature_model_path = Path(os.getenv(
            "DEEPSEIS_FEATURE_MODEL",
            "artifacts/models/mw40_pm60m_hhz_clean_gap48h_hist_gradient.joblib"
        ))

        manifest = load_manifest(splits_dir)
        experiment = manifest.get("name", splits_dir.name) if manifest else "canonical"

        threshold = float(os.getenv("DEEPSEIS_THRESHOLD", "nan"))
        feature_set = None
        if MODEL_BACKEND == "feature" or model_arch.startswith("feature"):
            MODEL_BACKEND = "feature"
            if not feature_model_path.exists():
                raise FileNotFoundError(f"Feature model not found: {feature_model_path}")
            artifact = joblib.load(feature_model_path)
            if isinstance(artifact, dict) and "model" in artifact:
                model = artifact["model"]
                feature_set = artifact.get("feature_set", "stft_summary_v1")
                if np.isnan(threshold):
                    threshold = float(artifact.get("threshold", 0.5))
            else:
                model = artifact
                feature_set = "stft_summary_v1"
                if np.isnan(threshold):
                    threshold = 0.5
            model_arch = os.getenv("DEEPSEIS_MODEL", "feature_hist_gradient").lower()
        else:
            MODEL_BACKEND = "pytorch"
            if np.isnan(threshold):
                threshold = 0.5
            model = build_model(model_arch)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

        x_test_path = resolve_x_test_path(splits_dir, manifest)
        y_test_path = splits_dir / "y_test.npy"
        if not x_test_path.exists():
            raise FileNotFoundError(f"X_test not found: {x_test_path}")
        if not y_test_path.exists():
            raise FileNotFoundError(f"y_test not found: {y_test_path}")

        X_test = np.load(str(x_test_path), mmap_mode="r")
        y_test = np.load(str(y_test_path), mmap_mode="r")
        MAX_INDEX = len(y_test)

        APP_STATE.update({
            "ready": True,
            "backend": MODEL_BACKEND,
            "model_arch": model_arch,
            "checkpoint": str(checkpoint_path) if MODEL_BACKEND == "pytorch" else None,
            "model_path": str(feature_model_path) if MODEL_BACKEND == "feature" else None,
            "splits_dir": str(splits_dir),
            "experiment": experiment,
            "threshold": threshold,
            "feature_set": feature_set,
            "error": None,
        })
        print(f"[DeepSeis API] Ready: backend={MODEL_BACKEND}, model={model_arch}, experiment={experiment}")
    except Exception as exc:
        APP_STATE.update({"ready": False, "error": str(exc)})
        print(f"[DeepSeis API] Startup error: {exc}")


init_system()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    return jsonify(APP_STATE)


@app.route("/reset_simulation", methods=["POST"])
def reset_simulation():
    global CURRENT_INDEX
    CURRENT_INDEX = 0
    return jsonify({"status": "success", "message": "Simulasyon sifirlandi."})


@app.route("/get_next_batch", methods=["GET"])
def get_next_batch():
    global CURRENT_INDEX
    if not APP_STATE["ready"]:
        return jsonify({"status": "error", "message": APP_STATE["error"] or "System is not ready."})
    if CURRENT_INDEX >= MAX_INDEX:
        return jsonify({"status": "error", "message": "Veri akisi sona erdi."})

    batch_size = 5
    end_idx = min(CURRENT_INDEX + batch_size, MAX_INDEX)
    x_batch = X_test[CURRENT_INDEX:end_idx]
    y_batch = y_test[CURRENT_INDEX:end_idx]

    if MODEL_BACKEND == "feature":
        features = stft_summary_features(x_batch)
        probs = model.predict_proba(features)[:, 1]
    else:
        x_t = torch.tensor(x_batch, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(x_t)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    threshold = APP_STATE["threshold"]
    data = []
    for i, prob in enumerate(probs):
        data.append({
            "index": CURRENT_INDEX + i,
            "anomaly_score": float(prob),
            "ground_truth": int(y_batch[i]),
            "prediction": int(prob >= threshold),
            "threshold": threshold,
        })

    CURRENT_INDEX = end_idx
    return jsonify({"status": "success", "data": data, "meta": APP_STATE})


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=int(os.getenv("PORT", "5000")))
