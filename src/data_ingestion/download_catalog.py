"""
DeepSeis - Deprem Kataloğu İndirme Modülü
==========================================

Bu modül, FDSN event servisi üzerinden deprem kataloğu verilerini indirmek
için kullanılmaktadır. Sonuçlar CSV formatında kaydedilir.

Kullanım:
    python -m src.data_ingestion.download_catalog
    python -m src.data_ingestion.download_catalog --test

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import kilometer2degrees

# ============================================================
# Loglama Konfigürasyonu
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DeepSeis.Catalog")


def load_config(config_path: str = "configs/station_config.yaml") -> dict:
    """
    YAML konfigürasyon dosyasını yükler.

    Parameters
    ----------
    config_path : str
        Konfigürasyon dosyası yolu.

    Returns
    -------
    dict
        Konfigürasyon parametreleri sözlüğü.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Konfigürasyon yüklendi: {config_path}")
    return config


def download_earthquake_catalog(config: dict, test_mode: bool = False) -> pd.DataFrame:
    """
    FDSN event servisi üzerinden deprem kataloğu indirir.

    Parameters
    ----------
    config : dict
        Konfigürasyon sözlüğü.
    test_mode : bool
        True ise sadece ilk 1 aylık veri indirir.

    Returns
    -------
    pd.DataFrame
        Deprem kataloğu DataFrame'i.
    """
    # Zaman aralığı
    start = UTCDateTime(config["time_range"]["start"])
    end = UTCDateTime(config["time_range"]["end"])

    if test_mode:
        # Test modunda sadece 1 aylık veri
        end = start + 30 * 86400
        logger.info("TEST MODU: Sadece 1 aylık katalog indirilecek.")

    # Katalog parametreleri
    cat_config = config["catalog"]
    min_mag = cat_config["min_magnitude"]
    lat = cat_config["latitude"]
    lon = cat_config["longitude"]
    max_radius_km = cat_config["max_radius_km"]
    max_radius_deg = kilometer2degrees(max_radius_km)

    # FDSN istemcisi (katalog için)
    catalog_source = config["data_source"]["catalog_client"]
    logger.info(f"Katalog kaynağı: {catalog_source}")
    client = Client(catalog_source)

    logger.info(f"Deprem kataloğu indiriliyor...")
    logger.info(f"  Zaman aralığı : {start.date} → {end.date}")
    logger.info(f"  Merkez konum  : {lat}°N, {lon}°E")
    logger.info(f"  Arama yarıçapı: {max_radius_km} km ({max_radius_deg:.2f}°)")
    logger.info(f"  Min büyüklük  : Mw ≥ {min_mag}")

    try:
        catalog = client.get_events(
            starttime=start,
            endtime=end,
            latitude=lat,
            longitude=lon,
            maxradius=max_radius_deg,
            minmagnitude=min_mag,
            orderby="time"
        )
    except FDSNNoDataException:
        logger.warning("Belirtilen parametrelerle deprem bulunamadı.")
        return pd.DataFrame()

    # Katalog verilerini DataFrame'e dönüştür
    events = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]

        events.append({
            "event_id": str(event.resource_id),
            "time": str(origin.time),
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "depth_km": origin.depth / 1000.0 if origin.depth else None,
            "magnitude": magnitude.mag,
            "magnitude_type": magnitude.magnitude_type,
            "region": str(event.event_descriptions[0].text)
                       if event.event_descriptions else "Bilinmiyor"
        })

    df = pd.DataFrame(events)

    # Zaman sütununu datetime'a dönüştür
    df["time"] = pd.to_datetime(df["time"])

    # Büyüklüğe göre sırala (büyükten küçüğe)
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(f"\nKatalog istatistikleri:")
    logger.info(f"  Toplam deprem sayısı   : {len(df)}")
    if len(df) > 0:
        logger.info(f"  Mw ≥ 4.0 deprem sayısı: {len(df[df['magnitude'] >= 4.0])}")
        logger.info(f"  Mw ≥ 4.5 deprem sayısı: {len(df[df['magnitude'] >= 4.5])}")
        logger.info(f"  Mw ≥ 5.0 deprem sayısı: {len(df[df['magnitude'] >= 5.0])}")
        logger.info(f"  En büyük deprem        : Mw {df['magnitude'].max():.1f}")
        logger.info(f"  Ortalama derinlik      : {df['depth_km'].mean():.1f} km")

    return df


def save_catalog(df: pd.DataFrame, config: dict):
    """
    Deprem kataloğunu CSV dosyasına kaydeder.

    Parameters
    ----------
    df : pd.DataFrame
        Deprem kataloğu DataFrame'i.
    config : dict
        Konfigürasyon sözlüğü.
    """
    output_dir = Path(config["paths"]["raw_catalog"])
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / "earthquake_catalog.csv"
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Katalog kaydedildi: {filepath.absolute()}")
    logger.info(f"  Dosya boyutu: {filepath.stat().st_size / 1024:.1f} KB")


# ============================================================
# Ana Çalıştırma
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSeis - Deprem Kataloğu İndirme Aracı"
    )
    parser.add_argument(
        "--config", type=str, default="configs/station_config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test modu: sadece 1 aylık katalog indir"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        df = download_earthquake_catalog(config, test_mode=args.test)

        if len(df) > 0:
            save_catalog(df, config)
            print("\n" + "=" * 60)
            print("İLK 10 DEPREM:")
            print("=" * 60)
            print(df.head(10).to_string(index=False))
        else:
            logger.warning("Kaydedilecek deprem verisi bulunamadı.")

    except KeyboardInterrupt:
        logger.info("İndirme kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"Kritik hata: {e}")
        sys.exit(1)
