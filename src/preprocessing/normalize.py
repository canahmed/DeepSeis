"""
DeepSeis - Normalizasyon Modülü
================================

Bu modül, sismik sinyal pencerelerine normalizasyon uygulamak için
kullanılmaktadır. Pencere bazlı z-score normalizasyonu temel yöntemdir.

Formül: x_norm = (x - μ) / σ

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from typing import Tuple

import numpy as np

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Preprocessing")


def zscore_normalize(windows: np.ndarray,
                     epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pencere bazlı z-score normalizasyonu uygular.

    Her pencere bağımsız olarak normalize edilir:
        x_norm = (x - μ) / (σ + ε)

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi, şekil: (num_windows, window_samples)
    epsilon : float
        Sıfıra bölme hatalarını önlemek için küçük değer.

    Returns
    -------
    normalized : np.ndarray
        Normalize edilmiş pencereler.
    means : np.ndarray
        Her pencerenin ortalaması (ters dönüşüm için).
    stds : np.ndarray
        Her pencerenin standart sapması (ters dönüşüm için).
    """
    if windows.size == 0:
        return windows, np.array([]), np.array([])

    # Her pencere için ortalama ve standart sapma
    means = np.mean(windows, axis=1, keepdims=True)
    stds = np.std(windows, axis=1, keepdims=True)

    # Normalizasyon
    normalized = (windows - means) / (stds + epsilon)

    logger.info(f"Z-score normalizasyonu uygulandı: "
                f"{windows.shape[0]} pencere, "
                f"ortalama μ aralığı: [{means.min():.4f}, {means.max():.4f}], "
                f"ortalama σ aralığı: [{stds.min():.4f}, {stds.max():.4f}]")

    return normalized, means.squeeze(), stds.squeeze()


def minmax_normalize(windows: np.ndarray,
                     feature_range: Tuple[float, float] = (0, 1),
                     epsilon: float = 1e-8) -> np.ndarray:
    """
    Pencere bazlı min-max normalizasyonu uygular.

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi.
    feature_range : tuple
        Hedef aralık (min, max).
    epsilon : float
        Sıfıra bölme hatalarını önlemek için küçük değer.

    Returns
    -------
    np.ndarray
        Normalize edilmiş pencereler.
    """
    if windows.size == 0:
        return windows

    min_val = np.min(windows, axis=1, keepdims=True)
    max_val = np.max(windows, axis=1, keepdims=True)

    a, b = feature_range
    normalized = a + (windows - min_val) * (b - a) / (max_val - min_val + epsilon)

    logger.info(f"Min-max normalizasyonu uygulandı: "
                f"aralık [{a}, {b}]")

    return normalized


def inverse_zscore(normalized: np.ndarray,
                   means: np.ndarray,
                   stds: np.ndarray) -> np.ndarray:
    """
    Z-score normalizasyonunu ters çevirir (orijinal ölçeğe döner).

    Parameters
    ----------
    normalized : np.ndarray
        Normalize edilmiş pencereler.
    means : np.ndarray
        Orijinal ortalamalar.
    stds : np.ndarray
        Orijinal standart sapmalar.

    Returns
    -------
    np.ndarray
        Orijinal ölçeğe dönüştürülmüş pencereler.
    """
    means = means.reshape(-1, 1)
    stds = stds.reshape(-1, 1)
    return normalized * stds + means


def normalize_windows(windows: np.ndarray,
                      method: str = "zscore") -> Tuple[np.ndarray, dict]:
    """
    Belirtilen yönteme göre pencere normalizasyonu uygular.

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi.
    method : str
        Normalizasyon yöntemi: "zscore" veya "minmax".

    Returns
    -------
    normalized : np.ndarray
        Normalize edilmiş pencereler.
    norm_params : dict
        Normalizasyon parametreleri (ters dönüşüm için).
    """
    if method == "zscore":
        normalized, means, stds = zscore_normalize(windows)
        norm_params = {"method": "zscore", "means": means, "stds": stds}
    elif method == "minmax":
        normalized = minmax_normalize(windows)
        norm_params = {"method": "minmax"}
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")

    return normalized, norm_params
