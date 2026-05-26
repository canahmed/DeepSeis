"""
DeepSeis - Toplu Ön İşleme Orkestrasyon Betiği
=================================================

Bu betik, 181 günlük MiniSEED verisini toplu olarak işler:
    MiniSEED → Filtre → Pencereleme → Normalizasyon → STFT

Resume desteği: Zaten işlenmiş günler atlanır.

Kullanım:
    python -m src.preprocessing.run_preprocessing
    python -m src.preprocessing.run_preprocessing --test  # 3 gün ile test

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml
from tqdm import tqdm
from obspy import read, UTCDateTime

# ============================================================
# Loglama
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DeepSeis.Preprocessing")


def process_single_day(mseed_file: Path,
                       config: dict,
                       output_windows_dir: Path,
                       output_stft_dir: Path) -> dict:
    """
    Tek bir günlük MiniSEED dosyasını uçtan uca işler.

    Returns
    -------
    dict
        İşleme istatistikleri.
    """
    date_str = mseed_file.stem.split(".")[-1]  # "2023-02-06"

    stats = {
        "date": date_str,
        "file": mseed_file.name,
        "status": "pending",
        "channels": {},
        "total_windows": 0,
        "total_spectrograms": 0
    }

    try:
        # 1. MiniSEED oku
        st = read(str(mseed_file))
        logger.debug(f"Okundu: {mseed_file.name} ({len(st)} trace)")

        # 2. Filtreleme
        fc = config["preprocessing"]["filter"]
        st.detrend("demean")
        st.detrend("linear")
        st.taper(max_percentage=0.01, type="cosine")
        st.filter(
            "bandpass",
            freqmin=fc["freqmin"],
            freqmax=fc["freqmax"],
            corners=fc["corners"],
            zerophase=fc["zerophase"]
        )

        # 3. Pencereleme + Normalizasyon + STFT (kanal bazlı)
        wc = config["preprocessing"]["windowing"]
        sc = config["spectral"]["stft"]
        window_length_sec = wc["window_length_sec"]
        overlap_ratio = wc["overlap_ratio"]
        min_data_ratio = config["preprocessing"]["quality"]["min_data_ratio"]

        for tr in st:
            channel = tr.stats.channel
            sampling_rate = tr.stats.sampling_rate
            window_samples = int(window_length_sec * sampling_rate)
            hop_samples = int(window_samples * (1 - overlap_ratio))

            data = tr.data.astype(np.float32)
            total_samples = len(data)

            if total_samples < window_samples:
                logger.warning(f"  {channel}: Trace çok kısa ({total_samples}), atlanıyor.")
                continue

            # Pencere oluştur
            num_windows = (total_samples - window_samples) // hop_samples + 1
            windows = []
            metadata = []

            for i in range(num_windows):
                start_idx = i * hop_samples
                end_idx = start_idx + window_samples
                w = data[start_idx:end_idx]

                # Kalite kontrolü
                if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                    continue
                non_zero = np.count_nonzero(w) / len(w)
                if non_zero < min_data_ratio:
                    continue

                windows.append(w)

                # Metadata
                w_start = tr.stats.starttime + (start_idx / sampling_rate)
                w_end = tr.stats.starttime + (end_idx / sampling_rate)
                metadata.append({
                    "window_index": len(windows) - 1,
                    "channel": channel,
                    "date": date_str,
                    "start_time": str(w_start),
                    "end_time": str(w_end),
                    "sampling_rate": sampling_rate
                })

            if not windows:
                continue

            windows_arr = np.array(windows, dtype=np.float32)

            # 4. Z-score normalizasyon
            means = np.mean(windows_arr, axis=1, keepdims=True)
            stds = np.std(windows_arr, axis=1, keepdims=True)
            norm_windows = (windows_arr - means) / (stds + 1e-8)

            # 5. STFT
            from scipy.signal import stft as scipy_stft
            spectrograms = []
            for w in norm_windows:
                f, t, Zxx = scipy_stft(
                    w, fs=sampling_rate,
                    window=sc["window"],
                    nperseg=sc["n_fft"],
                    noverlap=sc["n_fft"] - sc["hop_length"],
                    nfft=sc["n_fft"]
                )
                Sxx = np.log1p(np.abs(Zxx) ** 2)
                spectrograms.append(Sxx)

            spectrograms_arr = np.array(spectrograms, dtype=np.float32)

            # 6. Kaydet
            # Pencereler
            win_file = output_windows_dir / f"{date_str}_{channel}_windows.npy"
            np.save(str(win_file), norm_windows)

            # Spektrogramlar
            stft_file = output_stft_dir / f"{date_str}_{channel}_stft.npy"
            np.save(str(stft_file), spectrograms_arr)

            # Metadata
            meta_file = output_windows_dir / f"{date_str}_{channel}_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, indent=2, ensure_ascii=False)

            # Frekans ekseni (sadece ilk seferde kaydet)
            freq_file = output_stft_dir / "frequencies.npy"
            if not freq_file.exists():
                np.save(str(freq_file), f)

            stats["channels"][channel] = {
                "windows": len(windows),
                "spectrograms": spectrograms_arr.shape[0],
                "window_shape": list(norm_windows.shape),
                "stft_shape": list(spectrograms_arr.shape)
            }
            stats["total_windows"] += len(windows)
            stats["total_spectrograms"] += spectrograms_arr.shape[0]

        stats["status"] = "success"

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.error(f"Hata ({mseed_file.name}): {e}")

    return stats


def run_full_preprocessing(config_path: str = "configs/station_config.yaml",
                           test_mode: bool = False):
    """
    Tüm MiniSEED dosyalarını toplu işler.
    """
    # Config yükle
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mseed_dir = Path(config["paths"]["raw_mseed"])
    output_windows = Path(config["paths"]["processed_windows"])
    output_stft = Path(config["paths"]["processed_stft"])

    output_windows.mkdir(parents=True, exist_ok=True)
    output_stft.mkdir(parents=True, exist_ok=True)

    # MiniSEED dosyalarını bul
    mseed_files = sorted(mseed_dir.glob("*.mseed"))
    if test_mode:
        mseed_files = mseed_files[:3]
        logger.info(f"TEST MODU: Sadece {len(mseed_files)} dosya işlenecek.")

    logger.info(f"Toplam {len(mseed_files)} MiniSEED dosyası bulundu.")

    # Zaten işlenmiş günleri kontrol et (resume desteği)
    processed_dates = set()
    for f in output_windows.glob("*_HHZ_windows.npy"):
        date_part = f.stem.split("_HHZ")[0]
        processed_dates.add(date_part)

    remaining = [f for f in mseed_files
                 if f.stem.split(".")[-1] not in processed_dates]

    if processed_dates:
        logger.info(f"Zaten işlenmiş: {len(processed_dates)} gün, "
                     f"kalan: {len(remaining)} gün")

    # Toplu işleme
    all_stats = []
    total_windows = 0
    total_spectrograms = 0
    errors = 0

    for mseed_file in tqdm(remaining, desc="Ön işleme", unit="gün"):
        stats = process_single_day(mseed_file, config, output_windows, output_stft)
        all_stats.append(stats)

        if stats["status"] == "success":
            total_windows += stats["total_windows"]
            total_spectrograms += stats["total_spectrograms"]
        else:
            errors += 1

    # Özet rapor
    logger.info("=" * 60)
    logger.info("ÖN İŞLEME TAMAMLANDI")
    logger.info(f"  İşlenen gün       : {len(remaining) - errors}")
    logger.info(f"  Hatalı gün        : {errors}")
    logger.info(f"  Toplam pencere    : {total_windows:,}")
    logger.info(f"  Toplam spektrogram: {total_spectrograms:,}")
    logger.info("=" * 60)

    # İstatistikleri kaydet
    stats_file = Path(config["paths"]["reports"]) / "preprocessing_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_days": len(remaining),
            "errors": errors,
            "total_windows": total_windows,
            "total_spectrograms": total_spectrograms,
            "daily_stats": all_stats
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"İstatistikler kaydedildi: {stats_file}")

    return all_stats


# ============================================================
# Ana Çalıştırma
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeis Toplu Ön İşleme")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    parser.add_argument("--test", action="store_true", help="Test modu (3 gün)")
    args = parser.parse_args()

    run_full_preprocessing(config_path=args.config, test_mode=args.test)
