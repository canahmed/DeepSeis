"""
Create a label/split experiment without duplicating large X arrays.

The existing processed split arrays are several GB. For alternative labeling
experiments, this script writes new y_train/y_val/y_test files and a manifest
that points to the existing X_train/X_val/X_test arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "data" / "processed" / "windows"
STFT_DIR = ROOT / "data" / "processed" / "stft"
CATALOG_PATH = ROOT / "data" / "raw" / "catalog" / "earthquake_catalog.csv"
EXPERIMENTS_DIR = ROOT / "data" / "processed" / "experiments"
CANONICAL_SPLITS_DIR = WINDOWS_DIR / "splits"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_time(value: str) -> np.datetime64:
    normalized = value.replace("Z", "+00:00")
    if "+" in normalized:
        normalized = normalized.split("+", 1)[0]
    return np.datetime64(normalized)


def load_catalog(min_magnitude: float) -> list[np.datetime64]:
    events: list[np.datetime64] = []
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if float(row["magnitude"]) >= min_magnitude:
                events.append(parse_time(row["time"]).astype("datetime64[ns]"))
    return sorted(events)


def load_metadata(channel: str) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    metadata: list[dict[str, Any]] = []
    daily_counts: dict[str, int] = {}
    dates: list[str] = []

    for meta_path in sorted(WINDOWS_DIR.glob(f"*_{channel}_metadata.json")):
        date = meta_path.stem.split(f"_{channel}")[0]
        items = json.loads(meta_path.read_text(encoding="utf-8"))
        dates.append(date)
        daily_counts[date] = len(items)
        metadata.extend(items)

    if not metadata:
        raise FileNotFoundError(f"No metadata found for channel {channel}")
    return metadata, daily_counts, sorted(dates)


def create_labels(
    metadata: list[dict[str, Any]],
    event_times: list[np.datetime64],
    hours: int,
    mode: str,
    window_minutes: int,
) -> np.ndarray:
    centers = np.empty(len(metadata), dtype="datetime64[ns]")
    for idx, item in enumerate(metadata):
        start = parse_time(item["start_time"]).astype("datetime64[ns]")
        end = parse_time(item["end_time"]).astype("datetime64[ns]")
        centers[idx] = start + (end - start) / 2

    labels = np.zeros(len(metadata), dtype=np.int32)
    pre_delta = np.timedelta64(hours, "h")
    around_delta = np.timedelta64(window_minutes, "m")
    for event_time in event_times:
        if mode == "pre":
            labels[(centers >= event_time - pre_delta) & (centers <= event_time)] = 1
        elif mode == "around":
            labels[(centers >= event_time - around_delta) & (centers <= event_time + around_delta)] = 1
        else:
            raise ValueError(f"Unknown label mode: {mode}")
    return labels


def date_slices(dates: list[str], daily_counts: dict[str, int]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for date in dates:
        count = int(daily_counts[date])
        slices[date] = (offset, offset + count)
        offset += count
    return slices


def split_dates(dates: list[str], train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    total_days = len(dates)
    train_end = int(total_days * train_ratio)
    val_end = int(total_days * (train_ratio + val_ratio))
    return {
        "train": dates[:train_end],
        "val": dates[train_end:val_end],
        "test": dates[val_end:],
    }


def save_split_labels(
    labels: np.ndarray,
    dates: list[str],
    daily_counts: dict[str, int],
    splits: dict[str, list[str]],
    output_dir: Path,
) -> dict[str, Any]:
    slices = date_slices(dates, daily_counts)
    split_info: dict[str, Any] = {}

    for split_name, split_date_list in splits.items():
        chunks = []
        for date in split_date_list:
            start, end = slices[date]
            chunks.append(labels[start:end])
        y = np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
        np.save(output_dir / f"y_{split_name}.npy", y)
        positive = int(y.sum())
        split_info[split_name] = {
            "days": len(split_date_list),
            "samples": int(len(y)),
            "positive": positive,
            "negative": int(len(y) - positive),
            "positive_pct": round(positive / len(y) * 100, 3) if len(y) else 0.0,
            "date_range": f"{split_date_list[0]} -> {split_date_list[-1]}" if split_date_list else "",
            "x_path": rel(CANONICAL_SPLITS_DIR / f"X_{split_name}.npy"),
            "y_path": rel(output_dir / f"y_{split_name}.npy"),
        }
    return split_info


def validate_against_existing_x(split_info: dict[str, Any]) -> None:
    for split_name, info in split_info.items():
        x_path = ROOT / info["x_path"]
        if not x_path.exists():
            raise FileNotFoundError(f"Referenced X split not found: {x_path}")
        x_shape = np.load(x_path, mmap_mode="r").shape
        if int(x_shape[0]) != int(info["samples"]):
            raise ValueError(
                f"{split_name} X/y sample mismatch: X={x_shape[0]}, y={info['samples']}. "
                "Do not reuse canonical X splits for this experiment."
            )


def create_experiment(
    min_magnitude: float,
    hours: int,
    mode: str,
    window_minutes: int,
    channel: str,
    name: str | None,
    train_ratio: float,
    val_ratio: float,
) -> Path:
    channel = channel.upper()
    if name:
        experiment_name = name
    elif mode == "around":
        experiment_name = f"mw{str(min_magnitude).replace('.', '')}_pm{window_minutes}m_{channel.lower()}"
    else:
        experiment_name = f"mw{str(min_magnitude).replace('.', '')}_{hours}h_{channel.lower()}"
    output_dir = EXPERIMENTS_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, daily_counts, dates = load_metadata(channel)
    event_times = load_catalog(min_magnitude)
    labels = create_labels(metadata, event_times, hours, mode=mode, window_minutes=window_minutes)
    splits = split_dates(dates, train_ratio, val_ratio)
    split_info = save_split_labels(labels, dates, daily_counts, splits, output_dir)
    validate_against_existing_x(split_info)

    positive = int(labels.sum())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": experiment_name,
        "channel": channel,
        "min_magnitude": min_magnitude,
        "label_mode": mode,
        "pre_event_hours": hours,
        "event_window_minutes": window_minutes if mode == "around" else None,
        "total_windows": int(len(labels)),
        "positive": positive,
        "negative": int(len(labels) - positive),
        "positive_pct": round(positive / len(labels) * 100, 3) if len(labels) else 0.0,
        "catalog_file": rel(CATALOG_PATH),
        "canonical_x_splits_dir": rel(CANONICAL_SPLITS_DIR),
        "splits": split_info,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DeepSeis label experiment")
    parser.add_argument("--min-magnitude", type=float, default=4.5)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--mode", type=str, default="pre", choices=["pre", "around"])
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--channel", type=str, default="HHZ")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = create_experiment(
        min_magnitude=args.min_magnitude,
        hours=args.hours,
        mode=args.mode,
        window_minutes=args.window_minutes,
        channel=args.channel,
        name=args.name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Wrote experiment {rel(output_dir)}")


if __name__ == "__main__":
    main()
