"""
DeepSeis - MiniSEED Sismik Veri İndirme Modülü
===============================================

Bu modül, FDSN web servisi üzerinden sürekli sismik veri indirmek için
kullanılmaktadır. ObsPy kütüphanesi ile KOERI veya IRIS veri merkezlerinden
MiniSEED formatında günlük parçalar halinde veri indirilir.

Kullanım:
    python -m src.data_ingestion.download_mseed
    python -m src.data_ingestion.download_mseed --test  # Sadece 1 günlük test

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import timedelta
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from tqdm import tqdm

# ============================================================
# Loglama Konfigürasyonu
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DeepSeis.DataIngestion")


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


def create_fdsn_client(config: dict) -> Client:
    """
    FDSN istemcisi oluşturur. Birincil kaynak başarısız olursa yedek kaynağa geçer.

    Parameters
    ----------
    config : dict
        Konfigürasyon sözlüğü.

    Returns
    -------
    Client
        ObsPy FDSN istemcisi.
    """
    primary = config["data_source"]["fdsn_client"]
    fallback = config["data_source"]["fdsn_fallback"]

    try:
        client = Client(primary)
        logger.info(f"FDSN istemcisi oluşturuldu: {primary}")
        return client
    except Exception as e:
        logger.warning(f"{primary} bağlantısı başarısız: {e}")
        logger.info(f"Yedek kaynağa geçiliyor: {fallback}")
        client = Client(fallback)
        logger.info(f"FDSN istemcisi oluşturuldu: {fallback}")
        return client


def download_day(client: Client, config: dict, day_start: UTCDateTime,
                 output_dir: Path) -> bool:
    """
    Belirli bir gün için sismik veri indirir ve MiniSEED olarak kaydeder.

    Parameters
    ----------
    client : Client
        FDSN istemcisi.
    config : dict
        Konfigürasyon sözlüğü.
    day_start : UTCDateTime
        Günün başlangıç zamanı (UTC).
    output_dir : Path
        Çıktı dizini.

    Returns
    -------
    bool
        İndirme başarılıysa True, aksi halde False.
    """
    day_end = day_start + 86400  # 24 saat = 86400 saniye

    # İstasyon parametreleri
    network = config["station"]["network"]
    station = config["station"]["station"]
    channels = config["station"]["channels"]
    location = config["station"]["location"]

    # Dosya adı: NET.STA.YYYY-MM-DD.mseed
    date_str = day_start.strftime("%Y-%m-%d")
    filename = f"{network}.{station}.{date_str}.mseed"
    filepath = output_dir / filename

    # Daha önce indirilmişse atla
    if filepath.exists():
        logger.info(f"Zaten mevcut, atlanıyor: {filename}")
        return True

    try:
        # FDSN üzerinden dalga formu verisi çek
        stream = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channels,
            starttime=day_start,
            endtime=day_end
        )

        if len(stream) == 0:
            logger.warning(f"Boş veri: {date_str}")
            return False

        # MiniSEED olarak kaydet
        stream.write(str(filepath), format="MSEED")
        logger.info(f"İndirildi: {filename} ({len(stream)} trace, "
                     f"{sum(tr.stats.npts for tr in stream)} örnek)")
        return True

    except FDSNNoDataException:
        logger.warning(f"Veri bulunamadı: {date_str}")
        return False
    except FDSNException as e:
        logger.error(f"FDSN hatası ({date_str}): {e}")
        return False
    except Exception as e:
        logger.error(f"Beklenmeyen hata ({date_str}): {e}")
        return False


def download_all(config: dict, test_mode: bool = False):
    """
    Konfigürasyondaki tüm zaman aralığı için sismik veri indirir.

    Parameters
    ----------
    config : dict
        Konfigürasyon sözlüğü.
    test_mode : bool
        True ise sadece 1 günlük veri indirir.
    """
    # Zaman aralığı
    start = UTCDateTime(config["time_range"]["start"])
    end = UTCDateTime(config["time_range"]["end"])

    if test_mode:
        end = start + 86400  # Sadece 1 gün
        logger.info("TEST MODU: Sadece 1 günlük veri indirilecek.")

    # Çıktı dizini
    output_dir = Path(config["paths"]["raw_mseed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # FDSN istemcisi
    client = create_fdsn_client(config)

    # Toplam gün sayısı
    total_days = int((end - start) / 86400)
    logger.info(f"İndirme başlatılıyor: {start.date} → {end.date} ({total_days} gün)")
    logger.info(f"İstasyon: {config['station']['network']}.{config['station']['station']}")
    logger.info(f"Kanallar: {config['station']['channels']}")

    # Günlük indirme döngüsü
    success_count = 0
    fail_count = 0
    current = start

    pbar = tqdm(total=total_days, desc="Sismik veri indiriliyor", unit="gün")

    while current < end:
        success = download_day(client, config, current, output_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1

        current += 86400
        pbar.update(1)

    pbar.close()

    # Özet rapor
    logger.info("=" * 60)
    logger.info("İNDİRME RAPORU")
    logger.info(f"  Toplam gün     : {total_days}")
    logger.info(f"  Başarılı       : {success_count}")
    logger.info(f"  Başarısız      : {fail_count}")
    logger.info(f"  Başarı oranı   : {success_count/total_days*100:.1f}%")
    logger.info(f"  Kayıt dizini   : {output_dir.absolute()}")
    logger.info("=" * 60)


# ============================================================
# Ana Çalıştırma
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSeis - FDSN Sismik Veri İndirme Aracı"
    )
    parser.add_argument(
        "--config", type=str, default="configs/station_config.yaml",
        help="Konfigürasyon dosyası yolu"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test modu: sadece 1 günlük veri indir"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        download_all(config, test_mode=args.test)
    except KeyboardInterrupt:
        logger.info("İndirme kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"Kritik hata: {e}")
        sys.exit(1)
