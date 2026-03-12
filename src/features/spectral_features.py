"""
DeepSeis - Spektral Öznitelik Çıkarma Modülü
==============================================

Bu modül, sismik sinyal pencerelerinden STFT (Kısa Süreli Fourier Dönüşümü)
tabanlı spektral öznitelikler çıkarmak için kullanılmaktadır.

STFT Parametreleri (varsayılan):
    - FFT boyutu: 256
    - Hop uzunluğu: 64
    - Pencere fonksiyonu: Hann

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal as scipy_signal

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Features")


def compute_stft(window: np.ndarray,
                 sampling_rate: float = 100.0,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 window_type: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tek bir pencere için STFT hesaplar.

    Parameters
    ----------
    window : np.ndarray
        Sinyal penceresi, şekil: (num_samples,)
    sampling_rate : float
        Örnekleme hızı (Hz).
    n_fft : int
        FFT boyutu.
    hop_length : int
        Hop uzunluğu (örnekler arasındaki adım).
    window_type : str
        Pencere fonksiyonu tipi ("hann", "hamming", vb.).

    Returns
    -------
    frequencies : np.ndarray
        Frekans ekseni (Hz).
    times : np.ndarray
        Zaman ekseni (saniye).
    Sxx : np.ndarray
        Güç spektral yoğunluğu matrisi (spektrogram).
    """
    frequencies, times, Zxx = scipy_signal.stft(
        window,
        fs=sampling_rate,
        window=window_type,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        return_onesided=True
    )

    # Büyüklük spektrumu (dB cinsinden)
    Sxx = np.abs(Zxx) ** 2

    return frequencies, times, Sxx


def compute_stft_batch(windows: np.ndarray,
                       sampling_rate: float = 100.0,
                       n_fft: int = 256,
                       hop_length: int = 64,
                       window_type: str = "hann",
                       log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bir dizi pencere için toplu STFT hesaplar.

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi, şekil: (num_windows, window_samples)
    sampling_rate : float
        Örnekleme hızı (Hz).
    n_fft : int
        FFT boyutu.
    hop_length : int
        Hop uzunluğu.
    window_type : str
        Pencere fonksiyonu.
    log_scale : bool
        Logaritmik ölçek uygulansın mı.

    Returns
    -------
    frequencies : np.ndarray
        Frekans ekseni (Hz).
    times : np.ndarray
        Zaman ekseni (saniye).
    spectrograms : np.ndarray
        Spektrogramlar dizisi, şekil: (num_windows, num_freqs, num_times)
    """
    if windows.size == 0:
        return np.array([]), np.array([]), np.array([])

    spectrograms = []

    for i, window in enumerate(windows):
        freqs, times, Sxx = compute_stft(
            window, sampling_rate, n_fft, hop_length, window_type
        )

        if log_scale:
            # Log ölçekleme (dB benzeri, 0 koruması ile)
            Sxx = np.log1p(Sxx)

        spectrograms.append(Sxx)

    spectrograms = np.array(spectrograms, dtype=np.float32)

    logger.info(f"STFT hesaplandı: {spectrograms.shape[0]} pencere → "
                f"çıktı boyutu {spectrograms.shape}")
    logger.info(f"  Frekans çözünürlüğü : {freqs[1] - freqs[0]:.2f} Hz "
                f"({len(freqs)} frekans çubuğu)")
    logger.info(f"  Zaman çözünürlüğü   : {times[1] - times[0]:.4f} s "
                f"({len(times)} zaman adımı)")

    return freqs, times, spectrograms


def save_spectrograms(spectrograms: np.ndarray,
                      frequencies: np.ndarray,
                      output_dir: str,
                      channel: str = "BHZ",
                      date_str: str = ""):
    """
    Spektrogramları NumPy dosyalarına kaydeder.

    Parameters
    ----------
    spectrograms : np.ndarray
        Spektrogramlar dizisi.
    frequencies : np.ndarray
        Frekans ekseni.
    output_dir : str
        Çıktı dizini.
    channel : str
        Kanal adı.
    date_str : str
        Tarih etiketi.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{date_str}_" if date_str else ""

    # Spektrogramlar
    spec_file = out_path / f"{prefix}{channel}_spectrograms.npy"
    np.save(str(spec_file), spectrograms)

    # Frekans ekseni
    freq_file = out_path / f"{prefix}{channel}_frequencies.npy"
    np.save(str(freq_file), frequencies)

    logger.info(f"Spektrogramlar kaydedildi: {spec_file.name} "
                f"({spectrograms.shape})")


def compute_spectral_features(Sxx: np.ndarray,
                              frequencies: np.ndarray) -> dict:
    """
    Tek bir spektrogramdan özet spektral öznitelikler çıkarır.

    Parameters
    ----------
    Sxx : np.ndarray
        Güç spektral yoğunluğu matrisi.
    frequencies : np.ndarray
        Frekans ekseni.

    Returns
    -------
    dict
        Spektral öznitelikler sözlüğü.
    """
    # Zaman ortalamalı güç spektrumu
    mean_power = np.mean(Sxx, axis=1)

    # Toplam enerji
    total_energy = np.sum(Sxx)

    # Spektral ağırlık merkezi (centroid)
    spectral_centroid = np.sum(frequencies * mean_power) / (np.sum(mean_power) + 1e-8)

    # Spektral bant genişliği
    spectral_bandwidth = np.sqrt(
        np.sum(((frequencies - spectral_centroid) ** 2) * mean_power) /
        (np.sum(mean_power) + 1e-8)
    )

    # Dominant frekans
    dominant_freq = frequencies[np.argmax(mean_power)]

    # Frekans bantlarındaki enerji oranları
    low_band = np.sum(mean_power[(frequencies >= 1) & (frequencies < 5)])
    mid_band = np.sum(mean_power[(frequencies >= 5) & (frequencies < 15)])
    high_band = np.sum(mean_power[(frequencies >= 15) & (frequencies <= 45)])
    total = low_band + mid_band + high_band + 1e-8

    features = {
        "total_energy": float(total_energy),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "dominant_frequency": float(dominant_freq),
        "low_band_ratio": float(low_band / total),    # 1-5 Hz
        "mid_band_ratio": float(mid_band / total),    # 5-15 Hz
        "high_band_ratio": float(high_band / total),  # 15-45 Hz
    }

    return features
