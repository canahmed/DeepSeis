"""
DeepSeis channel consistency audit.

Checks processed metadata/STFT/window files by date and channel. This is useful
before changing the modeling pipeline because the current project has three
component files on disk, while labels and splits appear to use HHZ only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "data" / "processed" / "windows"
STFT_DIR = ROOT / "data" / "processed" / "stft"
REPORTS_DIR = ROOT / "artifacts" / "reports"
CHANNELS = ("HHZ", "HHN", "HHE")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_metadata_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return len(json.load(handle))


def npy_shape(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    return list(np.load(path, mmap_mode="r").shape)


def parse_date_channel(path: Path, suffix: str) -> tuple[str, str] | None:
    stem = path.name.removesuffix(suffix)
    if "_" not in stem:
        return None
    date, channel = stem.rsplit("_", 1)
    if channel not in CHANNELS:
        return None
    return date, channel


def build_audit() -> dict[str, Any]:
    by_date: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for meta_path in sorted(WINDOWS_DIR.glob("*_metadata.json")):
        parsed = parse_date_channel(meta_path, "_metadata.json")
        if not parsed:
            continue
        date, channel = parsed
        win_path = WINDOWS_DIR / f"{date}_{channel}_windows.npy"
        stft_path = STFT_DIR / f"{date}_{channel}_stft.npy"
        by_date[date][channel] = {
            "metadata_file": rel(meta_path),
            "metadata_count": load_metadata_count(meta_path),
            "window_shape": npy_shape(win_path),
            "stft_shape": npy_shape(stft_path),
            "window_file_exists": win_path.exists(),
            "stft_file_exists": stft_path.exists(),
        }

    channel_totals: dict[str, int] = {channel: 0 for channel in CHANNELS}
    complete_dates: list[str] = []
    aligned_dates: list[str] = []
    mismatched_dates: dict[str, dict[str, int | None]] = {}
    missing_by_date: dict[str, list[str]] = {}

    for date, channels in sorted(by_date.items()):
        missing = [channel for channel in CHANNELS if channel not in channels]
        if missing:
            missing_by_date[date] = missing

        counts: dict[str, int | None] = {}
        for channel in CHANNELS:
            count = channels.get(channel, {}).get("metadata_count")
            counts[channel] = int(count) if count is not None else None
            if count is not None:
                channel_totals[channel] += int(count)

        if not missing:
            complete_dates.append(date)
            unique_counts = {counts[channel] for channel in CHANNELS}
            if len(unique_counts) == 1:
                aligned_dates.append(date)
            else:
                mismatched_dates[date] = counts

    label_totals = {}
    for label_path in sorted(WINDOWS_DIR.glob("labels_*h.npy")):
        label_totals[label_path.stem] = int(np.load(label_path, mmap_mode="r").shape[0])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dates_total": len(by_date),
        "complete_three_component_dates": len(complete_dates),
        "aligned_three_component_dates": len(aligned_dates),
        "mismatched_three_component_dates": len(mismatched_dates),
        "dates_with_missing_components": len(missing_by_date),
        "channel_totals": channel_totals,
        "label_totals": label_totals,
        "sample_mismatched_dates": dict(list(mismatched_dates.items())[:20]),
        "sample_missing_dates": dict(list(missing_by_date.items())[:20]),
        "recommendation": build_recommendation(channel_totals, label_totals, mismatched_dates),
    }


def build_recommendation(
    channel_totals: dict[str, int],
    label_totals: dict[str, int],
    mismatched_dates: dict[str, dict[str, int | None]],
) -> list[str]:
    notes: list[str] = []
    hhz_total = channel_totals.get("HHZ", 0)
    if label_totals:
        label_total_values = set(label_totals.values())
        if label_total_values == {hhz_total}:
            notes.append("Current labels align exactly with HHZ metadata counts.")
    if mismatched_dates:
        notes.append(
            "Naively stacking HHZ/HHN/HHE by array index is unsafe because many dates have different valid-window counts."
        )
        notes.append(
            "For a three-component model, rebuild preprocessing around common time windows or align by metadata timestamps."
        )
    notes.append(
        "Near-term model improvements should either document HHZ-only training or create a new timestamp-aligned 3C dataset."
    )
    return notes


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# DeepSeis Channel Audit",
        "",
        f"Generated: `{audit['generated_at_utc']}`",
        "",
        "## Summary",
        f"- Dates with any processed metadata: {audit['dates_total']}",
        f"- Dates with all three components: {audit['complete_three_component_dates']}",
        f"- Dates with equal HHZ/HHN/HHE counts: {audit['aligned_three_component_dates']}",
        f"- Dates with mismatched component counts: {audit['mismatched_three_component_dates']}",
        f"- Dates with missing components: {audit['dates_with_missing_components']}",
        "",
        "## Channel Totals",
        "| Channel | Metadata windows |",
        "|---|---:|",
    ]
    for channel, total in audit["channel_totals"].items():
        lines.append(f"| `{channel}` | {total} |")

    lines.extend(["", "## Label Totals", "| Label file | Windows |", "|---|---:|"])
    for label_name, total in audit["label_totals"].items():
        lines.append(f"| `{label_name}` | {total} |")

    lines.extend(["", "## Sample Mismatched Dates", "| Date | HHZ | HHN | HHE |", "|---|---:|---:|---:|"])
    for date, counts in audit["sample_mismatched_dates"].items():
        lines.append(f"| `{date}` | {counts.get('HHZ')} | {counts.get('HHN')} | {counts.get('HHE')} |")
    if not audit["sample_mismatched_dates"]:
        lines.append("| n/a | n/a | n/a | n/a |")

    lines.extend(["", "## Recommendation"])
    for note in audit["recommendation"]:
        lines.append(f"- {note}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    json_path = REPORTS_DIR / "channel_audit.json"
    md_path = REPORTS_DIR / "channel_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, md_path)
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")


if __name__ == "__main__":
    main()
