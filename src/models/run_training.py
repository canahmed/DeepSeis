"""
DeepSeis - Model Eğitimi Orkestrasyon Betiği
==============================================

Tüm modellerin (LSTM, GRU, CNN, Transformer V1/V2) eğitim,
doğrulama ve test aşamalarını uçtan uca çalıştırır.

Kullanım:
    python -m src.models.run_training --model transformer_v2 --epochs 50

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.dataset import create_dataloaders
from src.models.trainer import Trainer
from src.models.lstm_model import build_lstm
from src.models.gru_model import build_gru
from src.models.cnn_baseline import build_cnn
from src.models.transformer_model import build_transformer
from src.models.transformer_v2 import build_transformer_v2, FocalLoss
from src.models.evaluate import full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeepSeis.RunTraining")


def f1_at_threshold(y_true, y_prob, threshold: float) -> float:
    """Belirli eşikte F1 skorunu hesaplar."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def find_best_f1_threshold(y_true, y_prob, max_alert_rate: float = 0.10):
    """Validation setinde F1'i maksimize eden, alarm oranı sınırlı karar eşiğini bulur."""
    thresholds = np.unique(y_prob)
    if len(thresholds) > 500:
        thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 501)))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        pred_rate = float((y_prob >= threshold).mean())
        if pred_rate > max_alert_rate:
            continue
        score = f1_at_threshold(y_true, y_prob, float(threshold))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    if best_f1 < 0:
        logger.warning("Alarm oranı kısıtına uyan eşik bulunamadı; threshold=0.5 kullanılacak.")
        best_f1 = f1_at_threshold(y_true, y_prob, best_threshold)

    logger.info(
        f"  Validation F1-optimal eşik: {best_threshold:.4f} "
        f"(F1={best_f1:.4f}, max_alert_rate={max_alert_rate:.2%})"
    )
    return best_threshold, best_f1


def experiment_name_from_splits(splits_dir: Path) -> str:
    manifest_path = splits_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("name", splits_dir.name)
    return "canonical"


def build_loss(loss_name: str, train_dataset, focal_alpha: float, focal_gamma: float):
    """Komut satırı seçimine göre kayıp fonksiyonu veya sınıf ağırlığı döndürür."""
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return None, None
    if loss_name == "weighted_ce":
        return None, train_dataset.get_class_weights()
    if loss_name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma), None
    raise ValueError(f"Bilinmeyen loss: {loss_name}")


def main(model_arch: str, epochs: int, batch_size: int, config_path: str,
         splits_dir_arg: str = None,
         balanced: bool = True,
         loss_name: str = "ce",
         run_suffix: str = None,
         lr: float = None,
         weight_decay: float = None,
         patience: int = None,
         focal_alpha: float = 0.75,
         focal_gamma: float = 2.0,
         max_alert_rate: float = 0.10):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits_dir = Path(splits_dir_arg) if splits_dir_arg else Path(config["paths"]["processed_windows"]) / "splits"
    if not splits_dir.exists():
        logger.error("Bölünmüş veriler bulunamadı! Lütfen önce split_dataset.py çalıştırın.")
        return

    figures_dir = config["paths"]["figures"]
    experiment_name = experiment_name_from_splits(splits_dir)
    run_name = f"{model_arch}_{experiment_name}" if experiment_name != "canonical" else model_arch
    if run_suffix:
        clean_suffix = run_suffix.strip().lower().replace(" ", "_")
        run_name = f"{run_name}_{clean_suffix}"

    # 1. Veri Yükleyicileri Hazırla
    logger.info("*" * 60)
    logger.info("VERİ YÜKLENİYOR...")
    logger.info("*" * 60)
    logger.info(f"Split dizini: {splits_dir}")
    logger.info(f"Deney adı: {experiment_name}")
    loaders = create_dataloaders(str(splits_dir), batch_size=batch_size, balanced=balanced)

    train_loader = loaders['train']['loader']
    val_loader = loaders['val']['loader']
    test_loader = loaders['test']['loader']

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cihaz: {device_str}")
    logger.info(f"Train samples: {loaders['train']['size']:,}")
    logger.info(f"Val samples  : {loaders['val']['size']:,}")
    logger.info(f"Test samples : {loaders['test']['size']:,}")
    logger.info(f"Train positive: {loaders['train']['positive']:,}")
    logger.info(f"Val positive  : {loaders['val']['positive']:,}")
    logger.info(f"Test positive : {loaders['test']['positive']:,}")
    if loaders["val"]["positive"] == 0 or loaders["test"]["positive"] == 0:
        raise ValueError(
            "Validation/test split pozitif örnek içermiyor. "
            "Bu deney güvenilir threshold ve metrik üretimi için uygun değil."
        )

    # 2. Model ve eğitim parametreleri
    is_v2 = (model_arch == "transformer_v2")

    if model_arch == "lstm":
        model = build_lstm(freq_bins=129, time_steps=95, num_layers=2)
    elif model_arch == "gru":
        model = build_gru(freq_bins=129, time_steps=95, num_layers=2)
    elif model_arch == "cnn":
        model = build_cnn(freq_bins=129, time_steps=95)
    elif model_arch == "transformer":
        model = build_transformer(freq_bins=129, time_steps=95, d_model=128, nhead=4, num_encoder_layers=2)
    elif model_arch == "transformer_v2":
        model = build_transformer_v2(
            freq_bins=129, time_steps=95,
            cnn_channels=64, d_model=256, nhead=8,
            num_encoder_layers=4, dim_feedforward=512,
            dropout=0.2
        )
    else:
        logger.error(f"Bilinmeyen model tipi: {model_arch}")
        return

    # 3. Eğitim parametreleri (V2 için güçlendirilmiş)
    if is_v2:
        train_lr = lr if lr is not None else 3e-4
        train_patience = patience if patience is not None else 15
        train_weight_decay = weight_decay if weight_decay is not None else 1e-3
    else:
        train_lr = lr if lr is not None else 1e-4
        train_patience = patience if patience is not None else 5
        train_weight_decay = weight_decay if weight_decay is not None else 1e-4

    custom_criterion, class_weights = build_loss(
        loss_name=loss_name,
        train_dataset=loaders["train"]["dataset"],
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    if balanced and loss_name in {"weighted_ce", "focal"}:
        logger.warning(
            "Balanced sampler ile weighted/focal loss birlikte kullanılıyor. "
            "Bu agresif azınlık sınıfı telafisi yapabilir; sonucu baseline ile karşılaştırın."
        )
    logger.info(
        f"Eğitim ayarları: balanced={balanced}, loss={loss_name}, "
        f"lr={train_lr}, weight_decay={train_weight_decay}, patience={train_patience}"
    )

    # 4. Trainer başlat ve eğit
    trainer = Trainer(model, device=device_str)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=train_lr,
        weight_decay=train_weight_decay,
        class_weights=class_weights,
        patience=train_patience,
        model_name=run_name,
        custom_criterion=custom_criterion,
        use_cosine_schedule=is_v2
    )

    # 5. Test Seti Değerlendirme
    logger.info("*" * 60)
    logger.info(f"{model_arch.upper()} — TEST SETİ DEĞERLENDİRMESİ")
    logger.info("*" * 60)

    y_val_true, _, y_val_prob = trainer.predict(val_loader)
    best_threshold, best_val_f1 = find_best_f1_threshold(
        y_val_true, y_val_prob, max_alert_rate=max_alert_rate
    )

    y_true, y_pred_default, y_prob = trainer.predict(test_loader)
    y_pred_threshold = (y_prob >= best_threshold).astype(int)

    # Standart değerlendirme (eşik=0.5)
    full_evaluation(
        y_true, y_pred_default, y_prob,
        model_name=f"{run_name.upper()}_ARGMAX",
        figures_dir=figures_dir,
        history=history,
        threshold=0.5,
        threshold_source="argmax_binary_softmax",
        experiment_name=experiment_name
    )

    # Validation setinden seçilen eşiği test setinde ayrıca raporla.
    full_evaluation(
        y_true, y_pred_threshold, y_prob,
        model_name=f"{run_name.upper()}_VALF1",
        figures_dir=figures_dir,
        history=None,
        threshold=best_threshold,
        threshold_source="validation_f1",
        validation_f1_at_threshold=best_val_f1,
        experiment_name=experiment_name
    )

    training_config_path = Path(config["paths"]["reports"]) / f"{run_name.upper()}_training_config.json"
    training_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_arch,
            "experiment": experiment_name,
            "splits_dir": str(splits_dir),
            "run_name": run_name,
            "run_suffix": run_suffix,
            "epochs": epochs,
            "batch_size": batch_size,
            "balanced_sampler": balanced,
            "loss": loss_name,
            "lr": train_lr,
            "weight_decay": train_weight_decay,
            "patience": train_patience,
            "focal_alpha": focal_alpha if loss_name == "focal" else None,
            "focal_gamma": focal_gamma if loss_name == "focal" else None,
            "max_alert_rate": max_alert_rate,
            "validation_f1_threshold": best_threshold,
            "validation_f1_at_threshold": best_val_f1,
        }, f, indent=2)

    logger.info(f"\nTüm süreç başarıyla tamamlandı! Raporlar: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeis Modellerini Eğit")
    parser.add_argument("--model", type=str,
                        choices=["lstm", "gru", "cnn", "transformer", "transformer_v2"],
                        required=True, help="Eğitilecek model mimarisi")
    parser.add_argument("--epochs", type=int, default=20, help="Maksimum Epoch")
    parser.add_argument("--batch", type=int, default=256, help="Batch Size")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    parser.add_argument("--splits-dir", type=str, default=None,
                        help="Alternatif split/experiment dizini (manifest destekli)")
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                        help="Train setinde WeightedRandomSampler kullan")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "weighted_ce", "focal"],
                        help="Kayıp fonksiyonu stratejisi")
    parser.add_argument("--run-suffix", type=str, default=None,
                        help="Checkpoint ve rapor adlarına eklenecek deney etiketi")
    parser.add_argument("--lr", type=float, default=None, help="Öğrenme oranı override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience override")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-alert-rate", type=float, default=0.10,
                        help="Validation threshold seçimi için izin verilen en yüksek alarm oranı")

    args = parser.parse_args()
    main(
        model_arch=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        config_path=args.config,
        splits_dir_arg=args.splits_dir,
        balanced=args.balanced,
        loss_name=args.loss,
        run_suffix=args.run_suffix,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        max_alert_rate=args.max_alert_rate
    )
