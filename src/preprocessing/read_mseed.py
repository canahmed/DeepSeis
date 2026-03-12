"""
DeepSeis - MiniSEED Veri Okuma Modülü
======================================

Bu modül, MiniSEED formatındaki sismik veri dosyalarını okumak, kanal yapısını
doğrulamak ve veri bütünlüğünü kontrol etmek için kullanılmaktadır.

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from obspy import read, Stream, UTCDateTime
from obspy.core.trace import Trace

# ============================================================
# Loglama
# ============================================================
logger = logging.getLogger("DeepSeis.Preprocessing")


def read_mseed_file(filepath: str) -> Stream:
    """
    Tek bir MiniSEED dosyasını okur.

    Parameters
    ----------
    filepath : str
        MiniSEED dosya yolu.

    Returns
    -------
    Stream
        ObsPy Stream nesnesi.

    Raises
    ------
    FileNotFoundError
        Dosya bulunamazsa.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"MiniSEED dosyası bulunamadı: {filepath}")

    stream = read(str(path))
    logger.info(f"Okundu: {filepath} ({len(stream)} trace)")
    return stream


def read_all_mseed(mseed_dir: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Stream:
    """
    Bir dizindeki tüm MiniSEED dosyalarını okuyup birleştirir.

    Parameters
    ----------
    mseed_dir : str
        MiniSEED dosyalarının bulunduğu dizin.
    start_date : str, optional
        Başlangıç tarihi filtreleme (YYYY-MM-DD).
    end_date : str, optional
        Bitiş tarihi filtreleme (YYYY-MM-DD).

    Returns
    -------
    Stream
        Birleştirilmiş ObsPy Stream nesnesi.
    """
    mseed_path = Path(mseed_dir)
    if not mseed_path.exists():
        raise FileNotFoundError(f"MiniSEED dizini bulunamadı: {mseed_dir}")

    # Tüm .mseed dosyalarını bul ve sırala
    mseed_files = sorted(mseed_path.glob("*.mseed"))
    if not mseed_files:
        logger.warning(f"MiniSEED dosyası bulunamadı: {mseed_dir}")
        return Stream()

    logger.info(f"Toplam {len(mseed_files)} MiniSEED dosyası bulundu.")

    # Birleşik stream oluştur
    combined_stream = Stream()
    for f in mseed_files:
        try:
            st = read(str(f))
            combined_stream += st
        except Exception as e:
            logger.warning(f"Okuma hatası, atlanıyor: {f.name} → {e}")

    # Zaman filtreleme
    if start_date:
        start_utc = UTCDateTime(start_date)
        combined_stream = combined_stream.slice(starttime=start_utc)
    if end_date:
        end_utc = UTCDateTime(end_date)
        combined_stream = combined_stream.slice(endtime=end_utc)

    # Trace'leri birleştir (boşlukları doldurarak)
    combined_stream.merge(method=1, fill_value="interpolate")

    logger.info(f"Birleştirilmiş stream: {len(combined_stream)} trace")
    return combined_stream


def validate_channels(stream: Stream,
                      expected_channels: List[str] = None) -> dict:
    """
    Stream'deki kanalları doğrular ve metadata çıkarır.

    Parameters
    ----------
    stream : Stream
        ObsPy Stream nesnesi.
    expected_channels : list of str, optional
        Beklenen kanal kodları (örn: ["BHZ", "BHN", "BHE"]).

    Returns
    -------
    dict
        Kanal doğrulama raporu.
    """
    if expected_channels is None:
        expected_channels = ["BHZ", "BHN", "BHE"]

    report = {
        "total_traces": len(stream),
        "channels_found": [],
        "channels_missing": [],
        "sampling_rates": {},
        "time_ranges": {},
        "data_points": {},
        "is_valid": True
    }

    found_channels = set()
    for trace in stream:
        channel = trace.stats.channel
        found_channels.add(channel)
        report["sampling_rates"][channel] = trace.stats.sampling_rate
        report["time_ranges"][channel] = {
            "start": str(trace.stats.starttime),
            "end": str(trace.stats.endtime),
            "duration_sec": trace.stats.endtime - trace.stats.starttime
        }
        report["data_points"][channel] = trace.stats.npts

    report["channels_found"] = sorted(list(found_channels))

    # Eksik kanalları kontrol et
    for ch in expected_channels:
        if ch not in found_channels:
            report["channels_missing"].append(ch)
            report["is_valid"] = False

    # Örnekleme hızı tutarlılığını kontrol et
    rates = list(report["sampling_rates"].values())
    if len(set(rates)) > 1:
        logger.warning(f"Farklı örnekleme hızları tespit edildi: {rates}")
        report["is_valid"] = False

    # Raporu logla
    logger.info("=" * 50)
    logger.info("KANAL DOĞRULAMA RAPORU")
    logger.info(f"  Toplam trace    : {report['total_traces']}")
    logger.info(f"  Bulunan kanallar: {report['channels_found']}")
    if report["channels_missing"]:
        logger.warning(f"  Eksik kanallar  : {report['channels_missing']}")
    for ch, rate in report["sampling_rates"].items():
        logger.info(f"  {ch} örnekleme hızı: {rate} Hz")
        logger.info(f"  {ch} veri noktası  : {report['data_points'][ch]:,}")
    logger.info(f"  Geçerli         : {'Evet' if report['is_valid'] else 'HAYIR'}")
    logger.info("=" * 50)

    return report


def get_stream_info(stream: Stream) -> dict:
    """
    Stream hakkında özet bilgi döndürür.

    Parameters
    ----------
    stream : Stream
        ObsPy Stream nesnesi.

    Returns
    -------
    dict
        Stream özet bilgisi.
    """
    if len(stream) == 0:
        return {"empty": True}

    info = {
        "network": stream[0].stats.network,
        "station": stream[0].stats.station,
        "num_traces": len(stream),
        "channels": sorted(list(set(tr.stats.channel for tr in stream))),
        "sampling_rate": stream[0].stats.sampling_rate,
        "start_time": str(min(tr.stats.starttime for tr in stream)),
        "end_time": str(max(tr.stats.endtime for tr in stream)),
        "total_samples": sum(tr.stats.npts for tr in stream),
        "total_duration_sec": max(tr.stats.endtime for tr in stream) -
                              min(tr.stats.starttime for tr in stream)
    }

    return info


# ============================================================
# Test Çalıştırma
# ============================================================
if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="MiniSEED Okuma Testi")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    parser.add_argument("--test", action="store_true", help="Test modu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mseed_dir = config["paths"]["raw_mseed"]

    try:
        stream = read_all_mseed(mseed_dir)
        if len(stream) > 0:
            report = validate_channels(stream)
            info = get_stream_info(stream)
            print("\nStream Bilgisi:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        else:
            print("MiniSEED verisi bulunamadı. Önce download_mseed.py çalıştırın.")
    except Exception as e:
        print(f"Hata: {e}")
