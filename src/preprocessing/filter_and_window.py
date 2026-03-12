"""
DeepSeis - Filtreleme ve Pencereleme Modülü
============================================

Bu modül, ham sismik sinyallere bant-geçiren filtre uygulamak ve kayan
pencere segmentasyonu gerçekleştirmek için kullanılmaktadır.

İşlem Akışı:
    1. Butterworth bant-geçiren filtre (1-45 Hz, 4. derece)
    2. Kayan pencere ile segmentasyon (60 sn, %50 örtüşme)
    3. Kalite kontrolü (bozuk segmentlerin elenmesi)
    4. Pencerelerin NumPy dizileri olarak kaydedilmesi

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from tqdm import tqdm

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Preprocessing")


def apply_bandpass_filter(stream: Stream,
                          freqmin: float = 1.0,
                          freqmax: float = 45.0,
                          corners: int = 4,
                          zerophase: bool = True) -> Stream:
    """
    Stream'e Butterworth bant-geçiren filtre uygular.

    Parameters
    ----------
    stream : Stream
        ObsPy Stream nesnesi.
    freqmin : float
        Alt kesim frekansı (Hz).
    freqmax : float
        Üst kesim frekansı (Hz).
    corners : int
        Filtre derecesi.
    zerophase : bool
        Sıfır fazlı filtreleme uygulansın mı.

    Returns
    -------
    Stream
        Filtrelenmiş Stream nesnesi.
    """
    # Orijinal stream'i kopyala (in-place değişiklik yapmamak için)
    filtered_stream = stream.copy()

    # Trend ve ortalamayı kaldır
    filtered_stream.detrend("demean")
    filtered_stream.detrend("linear")

    # Taper uygula (kenar etkilerini azaltmak için)
    filtered_stream.taper(max_percentage=0.01, type="cosine")

    # Butterworth bant-geçiren filtre
    filtered_stream.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=corners,
        zerophase=zerophase
    )

    logger.info(f"Butterworth bant-geçiren filtre uygulandı: "
                f"{freqmin}-{freqmax} Hz, derece={corners}, "
                f"sıfır-faz={'Evet' if zerophase else 'Hayır'}")

    return filtered_stream


def create_windows(trace: Trace,
                   window_length_sec: float = 60.0,
                   overlap_ratio: float = 0.5,
                   min_data_ratio: float = 0.95) -> Tuple[np.ndarray, List[dict]]:
    """
    Tekil bir trace'i kayan pencere ile segmentlere ayırır.

    Parameters
    ----------
    trace : Trace
        ObsPy Trace nesnesi.
    window_length_sec : float
        Pencere uzunluğu (saniye).
    overlap_ratio : float
        Örtüşme oranı (0.0 - 1.0).
    min_data_ratio : float
        Minimum veri doluluğu oranı (bu oranın altındaki pencereler elenir).

    Returns
    -------
    windows : np.ndarray
        Pencereler dizisi, şekil: (num_windows, window_samples)
    metadata : list of dict
        Her pencereye ait metadata listesi.
    """
    sampling_rate = trace.stats.sampling_rate
    window_samples = int(window_length_sec * sampling_rate)
    hop_samples = int(window_samples * (1 - overlap_ratio))

    data = trace.data.astype(np.float32)
    total_samples = len(data)

    if total_samples < window_samples:
        logger.warning(f"Trace çok kısa ({total_samples} örnek), "
                       f"en az {window_samples} örnek gerekli.")
        return np.array([]), []

    # Pencere sayısını hesapla
    num_windows = (total_samples - window_samples) // hop_samples + 1

    windows = []
    metadata = []

    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        window_data = data[start_idx:end_idx]

        # Kalite kontrolü: NaN veya inf kontrolü
        if np.any(np.isnan(window_data)) or np.any(np.isinf(window_data)):
            continue

        # Veri doluluğu kontrolü (sıfır olmayan değerlerin oranı)
        non_zero_ratio = np.count_nonzero(window_data) / len(window_data)
        if non_zero_ratio < min_data_ratio:
            continue

        # Pencere metadata'sı
        window_start_time = trace.stats.starttime + (start_idx / sampling_rate)
        window_end_time = trace.stats.starttime + (end_idx / sampling_rate)

        meta = {
            "window_index": len(windows),
            "channel": trace.stats.channel,
            "start_time": str(window_start_time),
            "end_time": str(window_end_time),
            "start_sample": start_idx,
            "end_sample": end_idx,
            "sampling_rate": sampling_rate,
            "num_samples": window_samples,
            "rms": float(np.sqrt(np.mean(window_data ** 2))),
            "max_amplitude": float(np.max(np.abs(window_data)))
        }

        windows.append(window_data)
        metadata.append(meta)

    windows_array = np.array(windows, dtype=np.float32) if windows else np.array([])

    logger.info(f"Kanal {trace.stats.channel}: "
                f"{num_windows} pencereden {len(windows)} geçerli pencere üretildi "
                f"({window_length_sec}s, {overlap_ratio*100:.0f}% örtüşme)")

    return windows_array, metadata


def process_stream_to_windows(stream: Stream,
                              config: dict) -> dict:
    """
    Filtrelenmiş bir stream'i tüm kanallar için pencerelere ayırır.

    Parameters
    ----------
    stream : Stream
        Filtrelenmiş ObsPy Stream nesnesi.
    config : dict
        Konfigürasyon sözlüğü.

    Returns
    -------
    dict
        Kanal bazlı pencereler ve metadata sözlüğü.
        Yapı: {"BHZ": {"windows": np.ndarray, "metadata": [...]}, ...}
    """
    windowing_cfg = config["preprocessing"]["windowing"]
    quality_cfg = config["preprocessing"]["quality"]

    window_length = windowing_cfg["window_length_sec"]
    overlap = windowing_cfg["overlap_ratio"]
    min_data_ratio = quality_cfg["min_data_ratio"]

    result = {}

    for trace in stream:
        channel = trace.stats.channel

        windows, metadata = create_windows(
            trace,
            window_length_sec=window_length,
            overlap_ratio=overlap,
            min_data_ratio=min_data_ratio
        )

        if len(windows) > 0:
            result[channel] = {
                "windows": windows,
                "metadata": metadata
            }

    return result


def save_windows(windows_dict: dict,
                 output_dir: str,
                 date_str: str = ""):
    """
    Pencereleri NumPy dosyalarına kaydeder.

    Parameters
    ----------
    windows_dict : dict
        Kanal bazlı pencereler sözlüğü.
    output_dir : str
        Çıktı dizini.
    date_str : str
        Tarih etiketi (dosya adında kullanılır).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for channel, data in windows_dict.items():
        windows = data["windows"]
        metadata = data["metadata"]

        # Pencere verileri
        prefix = f"{date_str}_" if date_str else ""
        windows_file = out_path / f"{prefix}{channel}_windows.npy"
        np.save(str(windows_file), windows)

        # Metadata
        import json
        meta_file = out_path / f"{prefix}{channel}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Kaydedildi: {windows_file.name} "
                     f"({windows.shape[0]} pencere × {windows.shape[1]} örnek)")


# ============================================================
# Test Çalıştırma
# ============================================================
if __name__ == "__main__":
    import yaml
    import argparse
    from src.preprocessing.read_mseed import read_all_mseed

    parser = argparse.ArgumentParser(description="Filtreleme ve Pencereleme Testi")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # MiniSEED oku
    stream = read_all_mseed(config["paths"]["raw_mseed"])

    if len(stream) > 0:
        # Filtreleme
        filter_cfg = config["preprocessing"]["filter"]
        filtered = apply_bandpass_filter(
            stream,
            freqmin=filter_cfg["freqmin"],
            freqmax=filter_cfg["freqmax"],
            corners=filter_cfg["corners"],
            zerophase=filter_cfg["zerophase"]
        )

        # Pencereleme
        windows_dict = process_stream_to_windows(filtered, config)

        # Kaydet
        save_windows(windows_dict, config["paths"]["processed_windows"])

        # Özet
        print("\nPENCERELEME ÖZETİ:")
        for ch, data in windows_dict.items():
            print(f"  {ch}: {data['windows'].shape}")
    else:
        print("Veri bulunamadı. Önce download_mseed.py çalıştırın.")
