"""
DeepSeis - Etiketleme Modülü
==============================

Bu modül, deprem kataloğu bilgilerini kullanarak her veri penceresine
ikili etiket (0: normal, 1: deprem öncesi anomali) atamak için
kullanılmaktadır.

Etiketleme mantığı:
    - Belirli bir büyüklüğün üzerindeki (Mw ≥ 3.5) depremlerden önceki
      T saat içindeki pencereler "pozitif" (1) olarak etiketlenir.
    - Diğer pencereler "negatif" (0) olarak etiketlenir.
    - Farklı T değerleri (3h, 6h, 12h, 24h) desteklenmektedir.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Utils")


def load_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Deprem kataloğunu CSV dosyasından yükler.

    Parameters
    ----------
    catalog_path : str
        Katalog CSV dosyası yolu.

    Returns
    -------
    pd.DataFrame
        Deprem kataloğu DataFrame'i.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Katalog dosyası bulunamadı: {catalog_path}")

    df = pd.read_csv(str(path))
    df["time"] = pd.to_datetime(df["time"])

    logger.info(f"Katalog yüklendi: {len(df)} deprem, {catalog_path}")
    return df


def label_windows(window_metadata: List[dict],
                  catalog_df: pd.DataFrame,
                  pre_event_hours: float = 6.0,
                  min_magnitude: float = 3.5) -> np.ndarray:
    """
    Her pencereye deprem öncesi etiket atar.

    Etiketleme kuralı:
        yi = 1  eğer pencere, Mw ≥ min_magnitude depremden önceki
                pre_event_hours saat içindeyse
        yi = 0  aksi halde

    Parameters
    ----------
    window_metadata : list of dict
        Pencere metadata listesi (her biri "start_time" ve "end_time" içermeli).
    catalog_df : pd.DataFrame
        Deprem kataloğu ("time" ve "magnitude" sütunları gerekli).
    pre_event_hours : float
        Deprem öncesi etiketleme penceresi (saat).
    min_magnitude : float
        Minimum deprem büyüklüğü eşiği.

    Returns
    -------
    np.ndarray
        Etiket dizisi, şekil: (num_windows,), dtype=int (0 veya 1).
    """
    if len(window_metadata) == 0:
        return np.array([], dtype=int)

    # Büyüklük filtreleme
    significant_quakes = catalog_df[
        catalog_df["magnitude"] >= min_magnitude
    ].copy()

    if len(significant_quakes) == 0:
        logger.warning(f"Mw ≥ {min_magnitude} deprem bulunamadı.")
        return np.zeros(len(window_metadata), dtype=int)

    # Etiketleme
    labels = np.zeros(len(window_metadata), dtype=int)
    pre_event_delta = timedelta(hours=pre_event_hours)

    # Her deprem için öncesindeki pencereleri işaretle
    event_matches = 0
    for _, quake in significant_quakes.iterrows():
        quake_time = quake["time"]
        window_start = quake_time - pre_event_delta
        window_end = quake_time  # Deprem anına kadar

        for i, meta in enumerate(window_metadata):
            # Pencere zamanını parse et
            w_start = pd.Timestamp(meta["start_time"])
            w_end = pd.Timestamp(meta["end_time"])

            # Pencere ortası deprem öncesi pencerede mi?
            w_center = w_start + (w_end - w_start) / 2

            if window_start <= w_center <= window_end:
                if labels[i] == 0:
                    event_matches += 1
                labels[i] = 1

    # İstatistikler
    n_positive = int(np.sum(labels))
    n_negative = len(labels) - n_positive
    ratio = n_positive / len(labels) * 100 if len(labels) > 0 else 0

    logger.info(f"Etiketleme tamamlandı (pencere: {pre_event_hours}h, "
                f"eşik: Mw ≥ {min_magnitude}):")
    logger.info(f"  Toplam pencere  : {len(labels)}")
    logger.info(f"  Pozitif (1)     : {n_positive} ({ratio:.2f}%)")
    logger.info(f"  Negatif (0)     : {n_negative} ({100-ratio:.2f}%)")
    logger.info(f"  Dengesizlik     : 1:{n_negative//(n_positive+1)}")
    logger.info(f"  Eşleşen deprem  : {len(significant_quakes)}")

    return labels


def create_label_report(window_metadata: List[dict],
                        catalog_df: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """
    Farklı zaman pencereleri için etiketleme raporu oluşturur.

    Parameters
    ----------
    window_metadata : list of dict
        Pencere metadata listesi.
    catalog_df : pd.DataFrame
        Deprem kataloğu.
    config : dict
        Konfigürasyon sözlüğü.

    Returns
    -------
    pd.DataFrame
        Etiketleme raporu.
    """
    pre_event_windows = config["labeling"]["pre_event_windows_hours"]
    min_mag = config["catalog"]["min_magnitude"]

    report_data = []
    for hours in pre_event_windows:
        labels = label_windows(
            window_metadata, catalog_df,
            pre_event_hours=hours,
            min_magnitude=min_mag
        )

        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos

        report_data.append({
            "pre_event_hours": hours,
            "total_windows": len(labels),
            "positive": n_pos,
            "negative": n_neg,
            "positive_ratio_pct": n_pos / len(labels) * 100 if len(labels) > 0 else 0,
            "imbalance_ratio": f"1:{n_neg // (n_pos + 1)}"
        })

    report_df = pd.DataFrame(report_data)

    logger.info("\nETİKETLEME RAPORU:")
    logger.info(report_df.to_string(index=False))

    return report_df


def save_labels(labels: np.ndarray,
                output_dir: str,
                channel: str = "BHZ",
                pre_event_hours: float = 6.0):
    """
    Etiketleri NumPy dosyasına kaydeder.

    Parameters
    ----------
    labels : np.ndarray
        Etiket dizisi.
    output_dir : str
        Çıktı dizini.
    channel : str
        Kanal adı.
    pre_event_hours : float
        Etiketleme penceresi (dosya adında kullanılır).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"{channel}_labels_{int(pre_event_hours)}h.npy"
    filepath = out_path / filename
    np.save(str(filepath), labels)

    logger.info(f"Etiketler kaydedildi: {filepath}")
