#!/usr/bin/env python3
"""Aggregate cnn_smoke_results.csv into a per-configuration summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

GROUP_FIELDS = [
    "model_path",
    "device",
    "provider",
    "disable_fallback",
    "profile_enabled",
    "batch",
    "repeat",
    "warmup",
    "input_shape",
]

WEIGHTED_METRICS = [
    "min_ms",
    "mean_ms",
    "max_ms",
    "std_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "throughput_items_per_sec",
]

AVERAGED_FIELDS = [
    "duration_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cnn_smoke_results.csv by duplicated run configuration."
    )
    parser.add_argument(
        "--csv-in",
        default="results/raw/cnn_smoke_test/cnn_smoke_results.csv",
        help="Input CSV produced by 01_extract_cnn_smoke_results.py",
    )
    parser.add_argument(
        "--csv-out",
        default="results/raw/cnn_smoke_test/cnn_smoke_results_summary.csv",
        help="Output summary CSV path",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def weighted_average(weighted_sum: float, weight_total: int) -> float | None:
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def plain_average(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return total / count


def make_group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in GROUP_FIELDS)


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], dict[str, object]] = {}

    for row in rows:
        key = make_group_key(row)
        if key not in groups:
            groups[key] = {
                "group_values": {field: row[field] for field in GROUP_FIELDS},
                "total_runs": 0,
                "success_runs": 0,
                "failed_runs": 0,
                "incomplete_runs": 0,
                "attempt_index_max": 0,
                "count_weight_total": 0,
                "count_sum": 0,
                "weighted_sums": defaultdict(float),
                "plain_sums": defaultdict(float),
            }

        group = groups[key]
        group["total_runs"] += 1

        attempt_index = parse_int(row["attempt_index"])
        if attempt_index is not None:
            group["attempt_index_max"] = max(group["attempt_index_max"], attempt_index)

        status = row["status"]
        if status == "success":
            group["success_runs"] += 1
            run_count = parse_int(row["count"])
            if run_count is not None:
                group["count_sum"] += run_count
                group["count_weight_total"] += run_count

            for field in WEIGHTED_METRICS:
                value = parse_float(row[field])
                if value is not None and run_count is not None:
                    group["weighted_sums"][field] += value * run_count

            for field in AVERAGED_FIELDS:
                value = parse_float(row[field])
                if value is not None:
                    group["plain_sums"][field] += value
        elif status == "failed":
            group["failed_runs"] += 1
        else:
            group["incomplete_runs"] += 1

    summary_rows: list[dict[str, object]] = []
    for _, group in sorted(groups.items(), key=lambda item: item[0]):
        success_runs = group["success_runs"]
        count_weight_total = group["count_weight_total"]

        summary_row: dict[str, object] = {
            **group["group_values"],
            "total_runs": group["total_runs"],
            "success_runs": success_runs,
            "failed_runs": group["failed_runs"],
            "incomplete_runs": group["incomplete_runs"],
            "attempt_count": group["attempt_index_max"],
            "count_sum": group["count_sum"] if success_runs else "",
            "count_avg_per_success_run": plain_average(group["count_sum"], success_runs)
            if success_runs
            else "",
        }

        for field in AVERAGED_FIELDS:
            value = plain_average(group["plain_sums"][field], success_runs)
            summary_row[field] = round(value, 4) if value is not None else ""

        for field in WEIGHTED_METRICS:
            value = weighted_average(group["weighted_sums"][field], count_weight_total)
            summary_row[field] = round(value, 4) if value is not None else ""

        summary_rows.append(summary_row)

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *GROUP_FIELDS,
        "total_runs",
        "success_runs",
        "failed_runs",
        "incomplete_runs",
        "attempt_count",
        "count_sum",
        "count_avg_per_success_run",
        *AVERAGED_FIELDS,
        *WEIGHTED_METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    csv_in = resolve_repo_path(args.csv_in)
    csv_out = resolve_repo_path(args.csv_out)

    with csv_in.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary_rows = summarize_rows(rows)
    write_csv(csv_out, summary_rows)

    print(f"Read {len(rows)} rows from {csv_in}")
    print(f"Wrote {len(summary_rows)} summary rows to {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
