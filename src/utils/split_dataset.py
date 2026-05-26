"""
DeepSeis - Veri Bölme Modülü
==============================

Kronolojik (zaman bazlı) train/validation/test bölmesi yapar.
Veri sızmasını önlemek için rastgele değil, tarih sırasına göre böler.

Kullanım:
    python -m src.utils.split_dataset

Yazar: Can Ahmedi Yaşar PARLAK
Tarih: 2026
"""

import logging
import json
import argparse
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeepSeis.Split")


def split_dataset(config_path: str = "configs/station_config.yaml",
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15,
                  label_hours: int = 6,
                  channel: str = None):
    """
    Kronolojik bölme: Train(%70) / Val(%15) / Test(%15)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    windows_dir = Path(config["paths"]["processed_windows"])
    stft_dir = Path(config["paths"]["processed_stft"])

    # Pencere sırasını yükle
    order_file = windows_dir / "window_order.json"
    if not order_file.exists():
        raise FileNotFoundError("window_order.json bulunamadı! Önce run_labeling çalıştırın.")

    with open(order_file, "r", encoding="utf-8") as f:
        order_data = json.load(f)

    dates = order_data["dates_sorted"]
    daily_counts = order_data["daily_counts"]
    total_days = len(dates)
    selected_channel = (channel or order_data.get("channel") or "HHZ").upper()

    # Kronolojik bölme noktaları
    train_end = int(total_days * train_ratio)
    val_end = int(total_days * (train_ratio + val_ratio))

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    logger.info(f"Toplam gün: {total_days}")
    logger.info(f"Kanal: {selected_channel}")
    logger.info(f"  Train : {len(train_dates)} gün ({train_dates[0]} → {train_dates[-1]})")
    logger.info(f"  Val   : {len(val_dates)} gün ({val_dates[0]} → {val_dates[-1]})")
    logger.info(f"  Test  : {len(test_dates)} gün ({test_dates[0]} → {test_dates[-1]})")

    # Etiketleri yükle
    labels_file = windows_dir / f"labels_{label_hours}h.npy"
    all_labels = np.load(str(labels_file))
    expected_total = sum(int(count) for count in daily_counts.values())
    if len(all_labels) != expected_total:
        raise ValueError(
            f"Etiket sayısı pencere sırasıyla uyuşmuyor: "
            f"labels={len(all_labels)}, window_order={expected_total}. "
            "run_labeling betiğini aynı kanal ve saat ayarıyla yeniden çalıştırın."
        )

    # Her split için STFT ve etiketleri birleştir
    splits = {"train": train_dates, "val": val_dates, "test": test_dates}
    split_info = {}

    for split_name, split_dates in splits.items():
        stft_list = []
        label_list = []
        window_offset = 0

        # Tarih sırasıyla offset hesapla
        for date_str in dates:
            count = daily_counts[date_str]
            if date_str in split_dates:
                # STFT yükle
                stft_file = stft_dir / f"{date_str}_{selected_channel}_stft.npy"
                if stft_file.exists():
                    stft_data = np.load(str(stft_file))
                    expected_count = int(count)
                    if stft_data.shape[0] != expected_count:
                        raise ValueError(
                            f"{stft_file} pencere sayısı metadata ile uyuşmuyor: "
                            f"stft={stft_data.shape[0]}, metadata={expected_count}"
                        )
                    stft_list.append(stft_data)
                    label_list.append(all_labels[window_offset:window_offset + count])
                else:
                    raise FileNotFoundError(f"STFT dosyası bulunamadı: {stft_file}")
            window_offset += count

        if stft_list:
            X = np.concatenate(stft_list, axis=0)
            y = np.concatenate(label_list, axis=0)
        else:
            X = np.array([])
            y = np.array([])

        # Kaydet
        output_dir = windows_dir / "splits"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(output_dir / f"X_{split_name}.npy"), X)
        np.save(str(output_dir / f"y_{split_name}.npy"), y)

        n_pos = int(np.sum(y)) if len(y) > 0 else 0
        n_neg = len(y) - n_pos

        split_info[split_name] = {
            "days": len(split_dates),
            "samples": len(y),
            "positive": n_pos,
            "negative": n_neg,
            "positive_pct": round(n_pos / len(y) * 100, 2) if len(y) > 0 else 0,
            "shape": list(X.shape) if X.size > 0 else [],
            "date_range": f"{split_dates[0]} → {split_dates[-1]}" if split_dates else ""
        }

        logger.info(f"  {split_name.upper()}: {X.shape} — "
                     f"Poz: {n_pos} ({split_info[split_name]['positive_pct']}%) / "
                     f"Neg: {n_neg}")

    # Rapor kaydet
    report_path = Path(config["paths"]["reports"]) / "split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "channel": selected_channel,
            "label_hours": label_hours,
            "splits": split_info
        }, f, indent=2)

    logger.info(f"\nBölme raporu kaydedildi: {report_path}")
    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeis Veri Bölme")
    parser.add_argument("--config", type=str, default="configs/station_config.yaml")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--channel", type=str, default=None,
                        help="Kullanılacak kanal. Verilmezse window_order.json içindeki kanal kullanılır.")
    args = parser.parse_args()

    split_dataset(config_path=args.config, label_hours=args.hours, channel=args.channel)
