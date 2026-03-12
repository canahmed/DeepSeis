"""
DeepSeis - Spektrogram Görselleştirme Modülü
==============================================

Bu modül, STFT tabanlı spektrogramları görselleştirmek için
kullanılmaktadır. Zaman-frekans gösterimi ile sismik sinyallerin
spektral içeriği analiz edilir.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Visualization")


def plot_spectrogram(Sxx: np.ndarray,
                     frequencies: np.ndarray,
                     times: np.ndarray = None,
                     channel: str = "BHZ",
                     title: str = None,
                     cmap: str = "inferno",
                     save_path: str = None,
                     show: bool = False) -> plt.Figure:
    """
    Tek bir pencerenin STFT spektrogramını çizer.

    Parameters
    ----------
    Sxx : np.ndarray
        Güç spektral yoğunluğu matrisi, şekil: (num_freqs, num_times)
    frequencies : np.ndarray
        Frekans ekseni (Hz).
    times : np.ndarray, optional
        Zaman ekseni (saniye).
    channel : str
        Kanal adı.
    title : str, optional
        Grafik başlığı.
    cmap : str
        Renk haritası.
    save_path : str, optional
        Kayıt dosya yolu.
    show : bool
        Ekranda göster.

    Returns
    -------
    matplotlib.figure.Figure
        Oluşturulan figür.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    if times is None:
        times = np.arange(Sxx.shape[1])

    # Spektrogram çizimi
    im = ax.pcolormesh(
        times, frequencies, Sxx,
        shading="gouraud",
        cmap=cmap
    )

    # Renk çubuğu
    cbar = fig.colorbar(im, ax=ax, label="Güç (log ölçek)", pad=0.02)

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Frekans (Hz)")
    ax.set_title(title or f"STFT Spektrogramı - {channel}")
    ax.set_ylim(0, frequencies[-1])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Spektrogram kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_spectrogram_with_waveform(window: np.ndarray,
                                   Sxx: np.ndarray,
                                   frequencies: np.ndarray,
                                   sampling_rate: float = 100.0,
                                   times: np.ndarray = None,
                                   channel: str = "BHZ",
                                   title: str = None,
                                   save_path: str = None,
                                   show: bool = False) -> plt.Figure:
    """
    Dalga formu ve spektrogramı birlikte çizer (üst üste).

    Parameters
    ----------
    window : np.ndarray
        Sinyal penceresi.
    Sxx : np.ndarray
        Güç spektral yoğunluğu matrisi.
    frequencies : np.ndarray
        Frekans ekseni.
    sampling_rate : float
        Örnekleme hızı.
    times : np.ndarray, optional
        Zaman ekseni.
    channel : str
        Kanal adı.
    title : str, optional
        Grafik başlığı.
    save_path : str
        Kayıt dosya yolu.
    show : bool
        Ekranda göster.

    Returns
    -------
    matplotlib.figure.Figure
        Oluşturulan figür.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    height_ratios=[1, 2],
                                    sharex=False)

    # --- Dalga Formu ---
    time_axis = np.arange(len(window)) / sampling_rate
    ax1.plot(time_axis, window, color="#2563eb", linewidth=0.5)
    ax1.fill_between(time_axis, window, alpha=0.1, color="#2563eb")
    ax1.set_ylabel("Genlik")
    ax1.set_title(f"Ham Dalga Formu - {channel}", fontsize=11, loc="left")
    ax1.set_xlim(time_axis[0], time_axis[-1])

    # --- Spektrogram ---
    if times is None:
        times = np.linspace(0, len(window) / sampling_rate, Sxx.shape[1])

    im = ax2.pcolormesh(
        times, frequencies, Sxx,
        shading="gouraud",
        cmap="inferno"
    )

    cbar = fig.colorbar(im, ax=ax2, label="Güç (log ölçek)", pad=0.02)
    ax2.set_xlabel("Zaman (saniye)")
    ax2.set_ylabel("Frekans (Hz)")
    ax2.set_title(f"STFT Spektrogramı - {channel}", fontsize=11, loc="left")
    ax2.set_ylim(0, frequencies[-1])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Dalga formu + spektrogram kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_spectrogram_grid(spectrograms: np.ndarray,
                          frequencies: np.ndarray,
                          num_display: int = 6,
                          channel: str = "BHZ",
                          save_path: str = None,
                          show: bool = False) -> plt.Figure:
    """
    Birden fazla spektrogramı ızgara düzeninde gösterir.

    Parameters
    ----------
    spectrograms : np.ndarray
        Spektrogramlar dizisi, şekil: (N, num_freqs, num_times)
    frequencies : np.ndarray
        Frekans ekseni.
    num_display : int
        Gösterilecek spektrogram sayısı.
    channel : str
        Kanal adı.
    save_path : str
        Kayıt dosya yolu.
    show : bool
        Ekranda göster.
    """
    n = min(num_display, len(spectrograms))
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    indices = np.linspace(0, len(spectrograms) - 1, n, dtype=int)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.pcolormesh(
            np.arange(spectrograms[idx].shape[1]),
            frequencies,
            spectrograms[idx],
            shading="gouraud",
            cmap="inferno"
        )
        ax.set_title(f"Pencere #{idx}", fontsize=9)
        ax.set_ylabel("Frekans (Hz)")
        ax.set_ylim(0, frequencies[-1])

    # Boş eksenleri gizle
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Spektrogram Izgarası - {channel} ({n} pencere)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Spektrogram ızgarası kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
