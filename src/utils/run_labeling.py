"""
DeepSeis - Toplu Etiketleme Betiği
====================================

Bu betik, işlenmiş pencere metadata'larını deprem kataloğuyla eşleştirerek
etiketler üretir.

Kullanım:
    python -m src.utils.run_labeling
    python -m src.utils.run_labeling --hours 3    # 3 saatlik pencere

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ============================================================
# Loglama
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DeepSeis.Labeling")


def run_labeling(config_path: str = "configs/station_config.yaml",
                 pre_event_hours: float = None,
                 channel: str = "HHZ"):
    """
    Tüm pencereleri deprem kataloğuyla etiketler.
    """
    # Config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if pre_event_hours is None:
        pre_event_hours = config["labeling"]["default_window_hours"]

    # Deprem kataloğunu yükle
    catalog_path = Path(config["paths"]["raw_catalog"]) / "earthquake_catalog.csv"
    catalog_df = pd.read_csv(str(catalog_path))
    catalog_df["time"] = pd.to_datetime(catalog_df["time"])

    min_mag = config["catalog"]["min_magnitude"]
    significant = catalog_df[catalog_df["magnitude"] >= min_mag].copy()
    logger.info(f"Katalog: {len(catalog_df)} deprem, "
                f"Mw >= {min_mag}: {len(significant)} deprem")

    channel = channel.upper()

    # Pencere metadata dosyalarını bul
    windows_dir = Path(config["paths"]["processed_windows"])
    meta_files = sorted(windows_dir.glob(f"*_{channel}_metadata.json"))

    if not meta_files:
        logger.error(f"{channel} metadata dosyası bulunamadı! Önce run_preprocessing çalıştırın.")
        return

    logger.info(f"Kanal: {channel}")
    logger.info(f"Toplam {len(meta_files)} gün için metadata bulundu.")

    # Tüm metadata'ları topla
    all_metadata = []
    daily_window_counts = {}

    for mf in tqdm(meta_files, desc="Metadata okuma"):
        with open(mf, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        date_str = mf.stem.split(f"_{channel}")[0]
        daily_window_counts[date_str] = len(meta_list)
        all_metadata.extend(meta_list)

    logger.info(f"Toplam pencere: {len(all_metadata)}")

    # Etiketleme
    pre_event_delta = pd.Timedelta(hours=pre_event_hours)
    labels = np.zeros(len(all_metadata), dtype=np.int32)

    matched_events = 0
    for _, quake in significant.iterrows():
        quake_time = quake["time"]
        window_start = quake_time - pre_event_delta
        window_end = quake_time

        for i, meta in enumerate(all_metadata):
            w_start = pd.Timestamp(meta["start_time"])
            w_end = pd.Timestamp(meta["end_time"])
            w_center = w_start + (w_end - w_start) / 2

            if window_start <= w_center <= window_end:
                if labels[i] == 0:
                    matched_events += 1
                labels[i] = 1

    # İstatistikler
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    ratio = n_pos / len(labels) * 100

    logger.info("=" * 60)
    logger.info("ETİKETLEME SONUÇLARI")
    logger.info(f"  Etiketleme penceresi: {pre_event_hours} saat")
    logger.info(f"  Toplam pencere      : {len(labels):,}")
    logger.info(f"  Pozitif (anomali)   : {n_pos:,} ({ratio:.2f}%)")
    logger.info(f"  Negatif (normal)    : {n_neg:,} ({100-ratio:.2f}%)")
    logger.info(f"  Dengesizlik oranı   : 1:{n_neg//(n_pos+1)}")
    logger.info(f"  Eşleşen deprem      : {len(significant)}")
    logger.info("=" * 60)

    # Kaydet
    labels_file = windows_dir / f"labels_{int(pre_event_hours)}h.npy"
    np.save(str(labels_file), labels)
    logger.info(f"Etiketler kaydedildi: {labels_file}")

    # Günlük metadata sıralamasını kaydet (split için)
    order_file = windows_dir / "window_order.json"
    with open(order_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_windows": len(all_metadata),
            "channel": channel,
            "daily_counts": daily_window_counts,
            "dates_sorted": sorted(daily_window_counts.keys())
        }, f, indent=2)
    logger.info(f"Pencere sırası kaydedildi: {order_file}")

    # Rapor tüm zaman pencereleri için
    report_data = []
    for hours in config["labeling"]["pre_event_windows_hours"]:
        temp_labels = np.zeros(len(all_metadata), dtype=np.int32)
        delta = pd.Timedelta(hours=hours)

        for _, quake in significant.iterrows():
            qt = quake["time"]
            ws = qt - delta

            for i, meta in enumerate(all_metadata):
                wc = pd.Timestamp(meta["start_time"]) + \
                     (pd.Timestamp(meta["end_time"]) - pd.Timestamp(meta["start_time"])) / 2
                if ws <= wc <= qt:
                    temp_labels[i] = 1

        tp = int(np.sum(temp_labels))
        tn = len(temp_labels) - tp

        report_data.append({
            "hours": hours,
            "positive": tp,
            "negative": tn,
            "positive_pct": round(tp / len(temp_labels) * 100, 2),
            "imbalance": f"1:{tn // (tp + 1)}"
        })

        # Tüm zaman penceresi etiketlerini kaydet
        np.save(str(windows_dir / f"labels_{int(hours)}h.npy"), temp_labels)

    report_df = pd.DataFrame(report_data)
    logger.info(f"\nZAMAN PENCERESİ KARŞILAŞTIRMASI:\n{report_df.to_string(index=False)}")

    # Rapor kaydet
    report_path = Path(config["paths"]["reports"]) / "labeling_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeis Etiketleme")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    parser.add_argument("--hours", type=float, default=None,
                        help="Deprem öncesi pencere (saat)")
    parser.add_argument("--channel", type=str, default="HHZ",
                        help="Etiketlenecek kanal (varsayılan: HHZ)")
    args = parser.parse_args()

    run_labeling(config_path=args.config, pre_event_hours=args.hours, channel=args.channel)
