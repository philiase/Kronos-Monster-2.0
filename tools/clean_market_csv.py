"""Clean broker/exported OHLCV CSV files into the Kronos Monster format.

The script accepts common MetaTrader-style tab-separated exports:

    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>

It also accepts files that are already close to the target format. Output columns:

    timestamps,open,high,low,close,volume,amount
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


TARGET_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


def normalize_header(value: str) -> str:
    return value.strip().lower().replace("<", "").replace(">", "")


def parse_float(value: str, default: float = 0.0) -> float:
    value = str(value).strip()
    if value == "":
        return default
    return float(value)


def parse_timestamp(row: dict[str, str]) -> datetime:
    if row.get("timestamps"):
        return datetime.fromisoformat(row["timestamps"].strip())

    date_value = row.get("date", "").strip()
    time_value = row.get("time", "00:00:00").strip()
    raw_value = f"{date_value} {time_value}".strip()

    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(raw_value, fmt)
        except ValueError:
            pass

    raise ValueError(f"Cannot parse timestamp from row: {row}")


def sniff_dialect(path: Path) -> csv.Dialect:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def read_rows(path: Path) -> Iterable[dict[str, str]]:
    dialect = sniff_dialect(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row")

        reader.fieldnames = [normalize_header(name) for name in reader.fieldnames]
        for row in reader:
            yield {normalize_header(key): value for key, value in row.items() if key is not None}


def clean_rows(path: Path) -> list[dict[str, str]]:
    cleaned: dict[datetime, dict[str, str]] = {}

    for raw in read_rows(path):
        timestamp = parse_timestamp(raw)
        open_price = parse_float(raw["open"])
        high_price = parse_float(raw["high"])
        low_price = parse_float(raw["low"])
        close_price = parse_float(raw["close"])

        if high_price < max(open_price, close_price, low_price):
            raise ValueError(f"Invalid OHLC at {timestamp}: high is below another price")
        if low_price > min(open_price, close_price, high_price):
            raise ValueError(f"Invalid OHLC at {timestamp}: low is above another price")

        volume = parse_float(raw.get("volume") or raw.get("tickvol") or raw.get("vol") or "0")
        if volume == 0 and raw.get("tickvol"):
            volume = parse_float(raw["tickvol"])
        amount = parse_float(raw.get("amount", ""), default=close_price * volume)

        cleaned[timestamp] = {
            "timestamps": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "open": f"{open_price:.10g}",
            "high": f"{high_price:.10g}",
            "low": f"{low_price:.10g}",
            "close": f"{close_price:.10g}",
            "volume": f"{volume:.10g}",
            "amount": f"{amount:.10g}",
        }

    return [cleaned[key] for key in sorted(cleaned)]


def interval_report(rows: list[dict[str, str]]) -> str:
    if len(rows) < 2:
        return "not enough rows to detect interval"

    timestamps = [datetime.fromisoformat(row["timestamps"]) for row in rows]
    diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    common = max(set(diffs), key=diffs.count)
    gaps = sum(1 for diff in diffs if diff != common)
    return f"common interval={common}, non-common gaps={gaps}"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OHLCV market data for Kronos Monster.")
    parser.add_argument("input", type=Path, help="Raw broker CSV export")
    parser.add_argument("output", type=Path, help="Clean output CSV path")
    args = parser.parse_args()

    rows = clean_rows(args.input)
    if not rows:
        raise SystemExit("No rows were written; input appears empty.")

    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"start={rows[0]['timestamps']}")
    print(f"end={rows[-1]['timestamps']}")
    print(interval_report(rows))


if __name__ == "__main__":
    main()
