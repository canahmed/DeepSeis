"""
Analyze DeepSeis labeling choices without rewriting existing labels.

The current labels use a fixed magnitude threshold and pre-event window. This
script evaluates a grid of magnitude thresholds and pre-event hours on the
current channel metadata so we can choose a defensible thesis experiment setup.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "data" / "processed" / "windows"
CATALOG_PATH = ROOT / "data" / "raw" / "catalog" / "earthquake_catalog.csv"
REPORTS_DIR = ROOT / "artifacts" / "reports"

HOURS_GRID = (1, 3, 6, 12, 24)
MAG_GRID = (3.5, 4.0, 4.5, 5.0)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_time(value: str) -> np.datetime64:
    normalized = value.replace("Z", "+00:00")
    if "+" in normalized:
        normalized = normalized.split("+", 1)[0]
    if normalized.endswith("000"):
        normalized = normalized
    return np.datetime64(normalized)


def load_window_centers(channel: str) -> tuple[np.ndarray, dict[str, int], list[str]]:
    centers: list[np.datetime64] = []
    daily_counts: dict[str, int] = {}
    dates: list[str] = []
    for meta_path in sorted(WINDOWS_DIR.glob(f"*_{channel}_metadata.json")):
        date = meta_path.stem.split(f"_{channel}")[0]
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        dates.append(date)
        daily_counts[date] = len(data)
        for item in data:
            start = parse_time(item["start_time"])
            end = parse_time(item["end_time"])
            centers.append(start + (end - start) / 2)
    return np.array(centers, dtype="datetime64[ns]"), daily_counts, sorted(dates)


def load_catalog() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with CATALOG_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            events.append({
                "time": parse_time(row["time"]),
                "magnitude": float(row["magnitude"]),
                "depth_km": float(row["depth_km"]) if row.get("depth_km") else None,
                "region": row.get("region", ""),
            })
    return events


def label_count_for_grid(
    centers: np.ndarray,
    events: list[dict[str, Any]],
    min_magnitude: float,
    hours: int,
    split_slices: dict[str, slice],
) -> dict[str, Any]:
    labels = np.zeros(len(centers), dtype=bool)
    delta = np.timedelta64(hours, "h")
    selected_events = [event for event in events if event["magnitude"] >= min_magnitude]

    matched_events = 0
    for event in selected_events:
        event_time = event["time"].astype("datetime64[ns]")
        mask = (centers >= event_time - delta) & (centers <= event_time)
        if bool(mask.any()):
            matched_events += 1
            labels |= mask

    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    split_stats = {}
    for split_name, split_slice in split_slices.items():
        split_labels = labels[split_slice]
        split_positive = int(split_labels.sum())
        split_total = int(len(split_labels))
        split_stats[split_name] = {
            "total": split_total,
            "positive": split_positive,
            "positive_pct": round(split_positive / split_total * 100, 3) if split_total else 0.0,
        }

    return {
        "min_magnitude": min_magnitude,
        "hours": hours,
        "events": len(selected_events),
        "matched_events": matched_events,
        "positive": positive,
        "negative": negative,
        "positive_pct": round(positive / len(labels) * 100, 3) if len(labels) else 0.0,
        "imbalance_negative_per_positive": round(negative / positive, 3) if positive else None,
        "splits": split_stats,
    }


def build_audit(channel: str) -> dict[str, Any]:
    channel = channel.upper()
    centers, daily_counts, dates = load_window_centers(channel)
    events = load_catalog()

    train_end_day = int(len(dates) * 0.70)
    val_end_day = int(len(dates) * 0.85)
    split_date_groups = {
        "train": dates[:train_end_day],
        "val": dates[train_end_day:val_end_day],
        "test": dates[val_end_day:],
    }
    split_slices = {}
    offset = 0
    date_slices = {}
    for date in dates:
        count = int(daily_counts[date])
        date_slices[date] = slice(offset, offset + count)
        offset += count

    split_slices = {
        split_name: slice(
            date_slices[split_dates[0]].start,
            date_slices[split_dates[-1]].stop,
        )
        for split_name, split_dates in split_date_groups.items()
        if split_dates
    }

    rows = []
    for min_magnitude in MAG_GRID:
        for hours in HOURS_GRID:
            rows.append(label_count_for_grid(centers, events, min_magnitude, hours, split_slices))

    recommended = [
        row for row in rows
        if (
            1.0 <= row["positive_pct"] <= 20.0
            and row["matched_events"] >= 5
            and all(split["positive"] > 0 for split in row["splits"].values())
        )
    ]
    recommended.sort(key=lambda row: (abs(row["positive_pct"] - 8.0), -row["matched_events"]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel": channel,
        "window_count": int(len(centers)),
        "catalog_file": rel(CATALOG_PATH),
        "hours_grid": list(HOURS_GRID),
        "magnitude_grid": list(MAG_GRID),
        "matrix": rows,
        "recommended_candidates": recommended[:5],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DeepSeis Labeling Matrix",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Channel: `{report['channel']}`",
        f"Window count: `{report['window_count']}`",
        "",
        "## Matrix",
        "| Min Mw | Hours | Events | Matched events | Positive | Negative | Positive % | Neg/Pos |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["matrix"]:
        imbalance = row["imbalance_negative_per_positive"]
        imbalance_text = "n/a" if imbalance is None else f"{imbalance:.3f}"
        lines.append(
            f"| {row['min_magnitude']:.1f} | {row['hours']} | {row['events']} | "
            f"{row['matched_events']} | {row['positive']} | {row['negative']} | "
            f"{row['positive_pct']:.3f} | {imbalance_text} |"
        )

    lines.extend([
        "",
        "## Split Distribution",
        "| Min Mw | Hours | Train +% | Val +% | Test +% | Train + | Val + | Test + |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["matrix"]:
        splits = row["splits"]
        lines.append(
            f"| {row['min_magnitude']:.1f} | {row['hours']} | "
            f"{splits['train']['positive_pct']:.3f} | {splits['val']['positive_pct']:.3f} | "
            f"{splits['test']['positive_pct']:.3f} | {splits['train']['positive']} | "
            f"{splits['val']['positive']} | {splits['test']['positive']} |"
        )

    lines.extend([
        "",
        "## Recommended Candidates",
        "| Min Mw | Hours | Matched events | Positive % | Val + | Test + | Reason |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ])
    if report["recommended_candidates"]:
        for row in report["recommended_candidates"]:
            lines.append(
                f"| {row['min_magnitude']:.1f} | {row['hours']} | {row['matched_events']} | "
                f"{row['positive_pct']:.3f} | {row['splits']['val']['positive']} | "
                f"{row['splits']['test']['positive']} | Rare enough for anomaly modeling and non-empty in each split. |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | No candidate met the heuristic. |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_audit(channel="HHZ")
    json_path = REPORTS_DIR / "labeling_matrix.json"
    md_path = REPORTS_DIR / "labeling_matrix.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")


if __name__ == "__main__":
    main()
