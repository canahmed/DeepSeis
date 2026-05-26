"""
DeepSeis - Jeofiziksel Zaman Çizelgesi Analizi
================================================

Eğitilmiş modelin (Transformer) test verisi üzerindeki
anomali tahmin skorlarını (y_prob) kronolojik olarak çizer
ve gerçek deprem anomali pencereleri (y_true) ile karşılaştırır.

Kullanım:
    python -m src.models.timeline_analysis --model transformer

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.transformer_model import build_transformer
from src.models.lstm_model import build_lstm
from src.models.gru_model import build_gru
from src.models.cnn_baseline import build_cnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeepSeis.Timeline")

def moving_average(a, n=120): # 120 pencere (yaklaşık 1 saatlik kayma)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main(model_arch: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits_dir = Path(config["paths"]["processed_windows"]) / "splits"
    x_test_path = splits_dir / "X_test.npy"
    y_test_path = splits_dir / "y_test.npy"

    if not x_test_path.exists():
        logger.error("Test verisi bulunamadı!")
        return

    logger.info("Veriler yükleniyor...")
    X_test = np.load(str(x_test_path))
    y_test = np.load(str(y_test_path))

    # Tensor yapısına çevir (batch, 1, 129, 95)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_test_t)
    loader = DataLoader(dataset, batch_size=512, shuffle=False) # KRONOLOJİK SIRA (Shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli Kur
    if model_arch == "lstm":
        model = build_lstm(freq_bins=129, time_steps=95, num_layers=2)
    elif model_arch == "gru":
        model = build_gru(freq_bins=129, time_steps=95, num_layers=2)
    elif model_arch == "cnn":
        model = build_cnn(freq_bins=129, time_steps=95)
    elif model_arch == "transformer":
        model = build_transformer(freq_bins=129, time_steps=95, d_model=128, nhead=4, num_encoder_layers=2)
    else:
        logger.error("Tanımsız model")
        return

    # Checkpoint yükle
    ckpt_path = Path("artifacts/checkpoints") / f"{model_arch}_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint bulunamadı: {ckpt_path}")
        return
    
    logger.info(f"Model yükleniyor: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Tahminler (Olasılıklar) üretiliyor...")
    y_probs = []
    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(device)
            logits = model(x_b)
            # Softmax ile [Normal(0), Anomali(1)]
            probs = F.softmax(logits, dim=1)[:, 1] 
            y_probs.extend(probs.cpu().numpy())
    
    y_probs = np.array(y_probs)

    # ==========================
    # GRAFİK ÇİZİMİ
    # ==========================
    logger.info("Grafik oluşturuluyor...")
    
    # Tamamını çizdirmek çok sıkışık olur, test setinden (21k) kesintisiz bir hafta (yaklaşık 10,000 pencere) seçelim
    # Veya ilk 6,000 pencere (yaklaşık 2-3 gün) detaylı görünüm için idealdir
    PLOT_LIMIT = 8000 
    
    y_prob_slice = y_probs[:PLOT_LIMIT]
    y_true_slice = y_test[:PLOT_LIMIT]
    
    # Çok gürültülü (hareketli) tahminleri düzleştirmek için hareketli ortalama
    # 60 pencere = yaklaşık 30 dakikalık aktivite (overlap nedeniyle tahmini)
    smoothed_probs = moving_average(y_prob_slice, n=60)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Zaman ekseni (Pencere indeksi)
    # Smoothed prob n=60 kaydığı için zaman ekseni ona uyarlanır
    x_axis = np.arange(len(smoothed_probs))
    
    # Orijinal tahmin
    ax.plot(x_axis, smoothed_probs, color='blue', linewidth=1.5, label='Model Anomali İhtimal Skoru (Smoothed)')
    
    # Zemin Rengi: Etiketlenen "Gerçek" Anomali zamanlarına kırmızı zemin bas (Deprem öncesi 6 saatlik periyotlar)
    # y_true_slice 1 ise kırmızı çizgi
    ax.fill_between(x_axis, 0, 1, where=(y_true_slice[59:PLOT_LIMIT] == 1), 
                    color='red', alpha=0.2, label='Gerçek Deprem Pencereleri (Katalog)')

    # %50 Karar Sınırı (Adil tahmin eşiği)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Klasik Karar Eşiği (0.5)')
    
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0, len(x_axis)])
    
    ax.set_title(f"DeepSeis - {model_arch.upper()} Deprem Anomali Skoru Zaman Çizelgesi (Test Seti Örneği)", fontsize=14)
    ax.set_xlabel("Zaman Serisi Adımları (Kronolojik Pencereler)", fontsize=12)
    ax.set_ylabel("Anomali İhtimali (Probability)", fontsize=12)
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Kaydet
    output_dir = Path("artifacts/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{model_arch.upper()}_timeline_analysis.png"
    
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    plt.close()
    
    logger.info(f"Jeofiziksel zaman çizelgesi grafiği başarıyla oluşturuldu: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "cnn", "transformer"])
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    args = parser.parse_args()
    
    main(args.model, args.config)
