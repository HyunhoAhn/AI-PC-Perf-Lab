#!/usr/bin/env python3
"""Extract structured CNN smoke test results from run_capture stdout.log."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import defaultdict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "raw" / "cnn_smoke_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse cnn_smoke_test stdout.log into a CSV summary."
    )
    parser.add_argument(
        "--stdout-log",
        default="results/raw/cnn_smoke_test/stdout.log",
        help="Path to run_capture stdout.log",
    )
    parser.add_argument(
        "--metadata-log",
        default="results/raw/cnn_smoke_test/metadata.jsonl",
        help="Optional path to run_capture metadata.jsonl",
    )
    parser.add_argument(
        "--csv-out",
        default="results/raw/cnn_smoke_test/cnn_smoke_results.csv",
        help="Path to output CSV",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_windows_path(raw_path: str) -> str:
    return raw_path.replace("/", "\\")


def parse_input_shape(raw_shape: str, batch: int | None) -> list[int]:
    dims = [int(part) for part in raw_shape.lower().split("x") if part]
    if batch is not None:
        return [batch] + dims
    return dims


def parse_command_config(command: str) -> dict[str, object]:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return {}

    config: dict[str, object] = {"disable_fallback": False, "profile_out": None}
    raw_input_shape: str | None = None

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "--model-path" and index + 1 < len(tokens):
            config["model_path"] = normalize_windows_path(tokens[index + 1]).lstrip(".\\")
            index += 2
            continue
        if token == "--device" and index + 1 < len(tokens):
            config["device"] = tokens[index + 1]
            index += 2
            continue
        if token == "--batch" and index + 1 < len(tokens):
            config["batch"] = int(tokens[index + 1])
            index += 2
            continue
        if token == "--repeat" and index + 1 < len(tokens):
            config["repeat"] = int(tokens[index + 1])
            index += 2
            continue
        if token == "--warmup" and index + 1 < len(tokens):
            config["warmup"] = int(tokens[index + 1])
            index += 2
            continue
        if token == "--input-shape" and index + 1 < len(tokens):
            raw_input_shape = tokens[index + 1]
            index += 2
            continue
        if token == "--profile-out" and index + 1 < len(tokens):
            config["profile_out"] = normalize_windows_path(tokens[index + 1])
            index += 2
            continue
        if token == "--disable-fallback":
            config["disable_fallback"] = True

        index += 1

    if raw_input_shape is not None:
        batch = config.get("batch")
        config["input_shape"] = parse_input_shape(
            raw_input_shape, batch if isinstance(batch, int) else None
        )

    return config


def merge_config(record: dict[str, object]) -> dict[str, object]:
    config: dict[str, object] = {}

    parsed_from_command = record.get("parsed_command_config", {})
    if isinstance(parsed_from_command, dict):
        config.update(parsed_from_command)

    raw_config = record.get("config", {})
    if isinstance(raw_config, dict):
        config.update(raw_config)

    return config


def load_metadata_by_timestamp(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    metadata: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            timestamp = entry.get("timestamp_utc")
            if isinstance(timestamp, str):
                metadata[timestamp] = entry
    return metadata


def make_row(
    record: dict[str, object],
    metadata_by_timestamp: dict[str, dict[str, object]],
    attempt_counters: dict[tuple[str | None, str | None, bool], int],
    run_index: int,
) -> dict[str, object]:
    config = merge_config(record)
    timing = record.get("timing", {})
    session = record.get("session", {})
    artifacts = record.get("artifacts", {})

    if not isinstance(timing, dict):
        timing = {}
    if not isinstance(session, dict):
        session = {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    timestamp_utc = record.get("timestamp_utc")
    metadata = (
        metadata_by_timestamp.get(timestamp_utc, {})
        if isinstance(timestamp_utc, str)
        else {}
    )
    if not isinstance(metadata, dict):
        metadata = {}

    profile_enabled = config.get("profile_out") is not None
    combo_key = (
        config.get("model_path") if isinstance(config.get("model_path"), str) else None,
        config.get("device") if isinstance(config.get("device"), str) else None,
        profile_enabled,
    )
    attempt_counters[combo_key] += 1

    exit_code = metadata.get("exit_code")
    has_timing = bool(timing)
    if isinstance(exit_code, int):
        status = "success" if exit_code == 0 else "failed"
    else:
        status = "success" if has_timing else "incomplete"

    row = {
        "run_index": run_index,
        "attempt_index": attempt_counters[combo_key],
        "status": status,
        "exit_code": exit_code,
        "timestamp_utc": record.get("timestamp_utc"),
        "finished_at_utc": metadata.get("finished_at_utc"),
        "duration_sec": metadata.get("duration_sec"),
        "command": record.get("command"),
        "model_path": config.get("model_path"),
        "device": config.get("device"),
        "provider": config.get("provider"),
        "disable_fallback": config.get("disable_fallback"),
        "profile_enabled": config.get("profile_out") is not None,
        "profile_out": config.get("profile_out"),
        "profile_artifact": artifacts.get("profile_out"),
        "requires_custom_ops": config.get("requires_custom_ops"),
        "custom_ops_registered": config.get("custom_ops_registered"),
        "batch": config.get("batch"),
        "repeat": config.get("repeat"),
        "warmup": config.get("warmup"),
        "input_shape": "x".join(str(dim) for dim in config.get("input_shape", [])),
        "input_names": ",".join(session.get("input_names", [])),
        "output_names": ",".join(session.get("output_names", [])),
        "count": timing.get("count"),
        "min_ms": timing.get("min_ms"),
        "mean_ms": timing.get("mean_ms"),
        "max_ms": timing.get("max_ms"),
        "std_ms": timing.get("std_ms"),
        "p50_ms": timing.get("p50_ms"),
        "p90_ms": timing.get("p90_ms"),
        "p95_ms": timing.get("p95_ms"),
        "throughput_items_per_sec": timing.get("throughput_items_per_sec"),
    }
    return row


def flush_record(records: list[dict[str, object]], record: dict[str, object]) -> None:
    if record:
        records.append(record.copy())


def parse_stdout_log(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line == "===== run_capture start =====":
                flush_record(records, current)
                current = {}
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key == "timestamp_utc":
                current[key] = value
                continue
            if key == "command":
                current[key] = value
                current["parsed_command_config"] = parse_command_config(value)
                continue

            if key in {"config", "session", "timing", "artifacts"}:
                current[key] = json.loads(value)

    flush_record(records, current)
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_index",
        "attempt_index",
        "status",
        "exit_code",
        "timestamp_utc",
        "finished_at_utc",
        "duration_sec",
        "command",
        "model_path",
        "device",
        "provider",
        "disable_fallback",
        "profile_enabled",
        "profile_out",
        "profile_artifact",
        "requires_custom_ops",
        "custom_ops_registered",
        "batch",
        "repeat",
        "warmup",
        "input_shape",
        "input_names",
        "output_names",
        "count",
        "min_ms",
        "mean_ms",
        "max_ms",
        "std_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "throughput_items_per_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    stdout_log = resolve_repo_path(args.stdout_log)
    metadata_log = resolve_repo_path(args.metadata_log)
    csv_out = resolve_repo_path(args.csv_out)

    records = parse_stdout_log(stdout_log)
    metadata_by_timestamp = load_metadata_by_timestamp(metadata_log)
    attempt_counters: dict[tuple[str | None, str | None, bool], int] = defaultdict(int)
    rows = [
        make_row(record, metadata_by_timestamp, attempt_counters, run_index)
        for run_index, record in enumerate(records, start=1)
    ]

    write_csv(csv_out, rows)
    print(f"Parsed {len(rows)} total runs")
    print(f"Wrote CSV: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
