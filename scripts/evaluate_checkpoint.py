"""
Evaluate a trained DeepSeis checkpoint on a split/experiment directory.

This is useful when threshold logic changes but retraining is not needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn_baseline import build_cnn
from src.models.dataset import create_dataloaders
from src.models.evaluate import full_evaluation
from src.models.gru_model import build_gru
from src.models.lstm_model import build_lstm
from src.models.transformer_model import build_transformer
from src.models.transformer_v2 import build_transformer_v2
from src.models.run_training import experiment_name_from_splits, find_best_f1_threshold


def build_model(model_arch: str):
    if model_arch == "cnn":
        return build_cnn(freq_bins=129, time_steps=95)
    if model_arch == "lstm":
        return build_lstm(freq_bins=129, time_steps=95, num_layers=2)
    if model_arch == "gru":
        return build_gru(freq_bins=129, time_steps=95, num_layers=2)
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


def predict(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            y_true.append(y_batch.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DeepSeis checkpoint")
    parser.add_argument("--model", required=True, choices=["cnn", "lstm", "gru", "transformer", "transformer_v2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-alert-rate", type=float, default=0.10)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    checkpoint_path = Path(args.checkpoint)
    experiment_name = experiment_name_from_splits(splits_dir)
    run_name = args.name or f"{args.model}_{experiment_name}_reeval"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loaders = create_dataloaders(str(splits_dir), batch_size=args.batch, balanced=False)
    y_val_true, _, y_val_prob = predict(model, loaders["val"]["loader"], device)
    threshold, val_f1 = find_best_f1_threshold(
        y_val_true, y_val_prob, max_alert_rate=args.max_alert_rate
    )

    y_test_true, y_test_argmax, y_test_prob = predict(model, loaders["test"]["loader"], device)
    y_test_threshold = (y_test_prob >= threshold).astype(int)

    full_evaluation(
        y_test_true, y_test_argmax, y_test_prob,
        model_name=f"{run_name.upper()}_ARGMAX",
        figures_dir=str(ROOT / "artifacts" / "figures"),
        threshold=0.5,
        threshold_source="argmax_binary_softmax",
        experiment_name=experiment_name,
    )
    full_evaluation(
        y_test_true, y_test_threshold, y_test_prob,
        model_name=f"{run_name.upper()}_VALF1_CONSTRAINED",
        figures_dir=str(ROOT / "artifacts" / "figures"),
        threshold=threshold,
        threshold_source="validation_f1_constrained",
        validation_f1_at_threshold=val_f1,
        experiment_name=experiment_name,
    )

    config_path = ROOT / "artifacts" / "reports" / f"{run_name.upper()}_reeval_config.json"
    config_path.write_text(json.dumps({
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "splits_dir": str(splits_dir),
        "experiment": experiment_name,
        "threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "max_alert_rate": args.max_alert_rate,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
