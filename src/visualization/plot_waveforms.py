"""
DeepSeis - Dalga Formu Görselleştirme Modülü
==============================================

Bu modül, sismik sinyal pencerelerinin ham dalga formlarını
görselleştirmek için kullanılmaktadır.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalışmak için
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Visualization")

# Profesyonel stil ayarları
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#dee2e6",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#adb5bd",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})


def plot_single_waveform(window: np.ndarray,
                         sampling_rate: float = 100.0,
                         channel: str = "BHZ",
                         title: str = None,
                         save_path: str = None,
                         show: bool = False) -> plt.Figure:
    """
    Tek bir pencerenin dalga formunu çizer.

    Parameters
    ----------
    window : np.ndarray
        Sinyal penceresi.
    sampling_rate : float
        Örnekleme hızı (Hz).
    channel : str
        Kanal adı.
    title : str, optional
        Grafik başlığı.
    save_path : str, optional
        Kayıt dosya yolu.
    show : bool
        Grafiği ekranda göster.

    Returns
    -------
    matplotlib.figure.Figure
        Oluşturulan figür.
    """
    time_axis = np.arange(len(window)) / sampling_rate

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(time_axis, window, color="#2563eb", linewidth=0.5, alpha=0.9)
    ax.fill_between(time_axis, window, alpha=0.1, color="#2563eb")

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Genlik")
    ax.set_title(title or f"Dalga Formu - {channel}")
    ax.set_xlim(time_axis[0], time_axis[-1])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Grafik kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_three_component(windows: dict,
                         sampling_rate: float = 100.0,
                         window_index: int = 0,
                         title: str = None,
                         save_path: str = None,
                         show: bool = False) -> plt.Figure:
    """
    Üç bileşenli (Z, N, E) dalga formlarını alt alta çizer.

    Parameters
    ----------
    windows : dict
        Kanal bazlı pencereler sözlüğü.
        Yapı: {"BHZ": np.ndarray, "BHN": np.ndarray, "BHE": np.ndarray}
    sampling_rate : float
        Örnekleme hızı.
    window_index : int
        Çizilecek pencere indeksi.
    title : str, optional
        Grafik başlığı.
    save_path : str, optional
        Kayıt dosya yolu.
    show : bool
        Ekranda göster.

    Returns
    -------
    matplotlib.figure.Figure
        Oluşturulan figür.
    """
    channels = sorted(windows.keys())
    n_channels = len(channels)

    colors = {"BHZ": "#dc2626", "BHN": "#2563eb", "BHE": "#16a34a",
              "HHZ": "#dc2626", "HHN": "#2563eb", "HHE": "#16a34a"}

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ax, channel in zip(axes, channels):
        data = windows[channel]
        if data.ndim == 2:
            data = data[window_index]

        time_axis = np.arange(len(data)) / sampling_rate
        color = colors.get(channel, "#6b7280")

        ax.plot(time_axis, data, color=color, linewidth=0.5, alpha=0.9)
        ax.fill_between(time_axis, data, alpha=0.08, color=color)
        ax.set_ylabel(f"{channel}\nGenlik")
        ax.set_title(f"Kanal: {channel}", fontsize=10, loc="left")
        ax.set_xlim(time_axis[0], time_axis[-1])

    axes[-1].set_xlabel("Zaman (saniye)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"3-bileşenli grafik kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_waveform_overview(windows: np.ndarray,
                           sampling_rate: float = 100.0,
                           num_windows: int = 5,
                           channel: str = "BHZ",
                           save_path: str = None,
                           show: bool = False) -> plt.Figure:
    """
    Birden fazla pencerenin dalga formlarını genel bakış olarak çizer.

    Parameters
    ----------
    windows : np.ndarray
        Pencereler dizisi, şekil: (num_windows, window_samples)
    sampling_rate : float
        Örnekleme hızı.
    num_windows : int
        Gösterilecek pencere sayısı.
    channel : str
        Kanal adı.
    save_path : str, optional
        Kayıt dosya yolu.
    show : bool
        Ekranda göster.
    """
    n = min(num_windows, len(windows))
    indices = np.linspace(0, len(windows) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        data = windows[idx]
        time_axis = np.arange(len(data)) / sampling_rate

        ax.plot(time_axis, data, color="#2563eb", linewidth=0.4)
        ax.set_ylabel(f"Pencere\n#{idx}")
        ax.set_xlim(time_axis[0], time_axis[-1])

    axes[-1].set_xlabel("Zaman (saniye)")
    fig.suptitle(f"Dalga Formu Genel Bakış - {channel} ({n} pencere)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Genel bakış grafiği kaydedildi: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
