"""
DeepSeis - Zaman Alanı Öznitelik Çıkarma Modülü
=================================================

Bu modül, sismik sinyal pencerelerinden temel zaman alanı öznitelikleri
çıkarmak için kullanılmaktadır.

Çıkarılan öznitelikler:
    - RMS (Root Mean Square)
    - Peak-to-Peak Genlik
    - Enerji
    - Sıfır Geçiş Sayısı
    - Basıklık (Kurtosis)
    - Çarpıklık (Skewness)

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from typing import List, Dict

import numpy as np
from scipy import stats as sp_stats

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Features")


def compute_time_features(window: np.ndarray) -> dict:
    """
    Tek bir pencere için zaman alanı öznitelikleri hesaplar.

    Parameters
    ----------
    window : np.ndarray
        Sinyal penceresi, şekil: (num_samples,)

    Returns
    -------
    dict
        Öznitelik sözlüğü.
    """
    # Temel istatistikler
    rms = float(np.sqrt(np.mean(window ** 2)))
    peak_to_peak = float(np.max(window) - np.min(window))
    energy = float(np.sum(window ** 2))
    mean_abs = float(np.mean(np.abs(window)))
    max_amplitude = float(np.max(np.abs(window)))
    variance = float(np.var(window))

    # Sıfır geçiş sayısı
    zero_crossings = int(np.sum(np.diff(np.sign(window)) != 0))

    # İstatistiksel momentler
    kurtosis = float(sp_stats.kurtosis(window))
    skewness = float(sp_stats.skew(window))

    # Crest faktörü (peak / RMS oranı)
    crest_factor = float(max_amplitude / (rms + 1e-8))

    features = {
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "energy": energy,
        "mean_abs_amplitude": mean_abs,
        "max_amplitude": max_amplitude,
        "variance": variance,
        "zero_crossing_rate": zero_crossings,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": crest_factor
    }

    return features


def compute_time_features_batch(windows: np.ndarray) -> List[dict]:
    """
    Bir dizi pencere için toplu zaman alanı öznitelikleri hesaplar.

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi, şekil: (num_windows, window_samples)

    Returns
    -------
    list of dict
        Her pencere için öznitelik sözlükleri listesi.
    """
    if windows.size == 0:
        return []

    features_list = []
    for window in windows:
        features = compute_time_features(window)
        features_list.append(features)

    logger.info(f"Zaman alanı öznitelikleri hesaplandı: "
                f"{len(features_list)} pencere, "
                f"{len(features_list[0])} öznitelik/pencere")

    return features_list


def features_to_array(features_list: List[dict]) -> np.ndarray:
    """
    Öznitelik sözlük listesini NumPy dizisine dönüştürür.

    Parameters
    ----------
    features_list : list of dict
        Öznitelik sözlükleri listesi.

    Returns
    -------
    np.ndarray
        Öznitelik matrisi, şekil: (num_windows, num_features)
    """
    if not features_list:
        return np.array([])

    keys = list(features_list[0].keys())
    array = np.array([[f[k] for k in keys] for f in features_list], dtype=np.float32)

    return array


def get_feature_names() -> List[str]:
    """
    Öznitelik adlarını döndürür.

    Returns
    -------
    list of str
        Öznitelik adları listesi.
    """
    return [
        "rms", "peak_to_peak", "energy", "mean_abs_amplitude",
        "max_amplitude", "variance", "zero_crossing_rate",
        "kurtosis", "skewness", "crest_factor"
    ]
