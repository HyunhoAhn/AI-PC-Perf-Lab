#!/usr/bin/env python3
"""Run an arbitrary command and capture raw stdout/stderr logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_log_marker(
    handle,
    *,
    timestamp: str,
    command_str: str,
    prepend_newline: bool,
) -> None:
    lines = []
    if prepend_newline:
        lines.append("")
    lines.append("===== run_capture start =====")
    lines.append(f"timestamp_utc={timestamp}")
    lines.append(f"command={command_str}")
    lines.append("")
    marker = "\n".join(lines).encode("utf-8", errors="replace")
    handle.write(marker)
    handle.flush()


def _migrate_legacy_metadata(metadata_path: Path, metadata_jsonl_path: Path) -> None:
    if not metadata_path.exists() or metadata_jsonl_path.exists():
        return

    try:
        legacy_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(legacy_payload, dict):
        return

    with metadata_jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(legacy_payload, sort_keys=True))
        handle.write("\n")


def _read_manifest_mode(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.yml"
    if not manifest_path.exists():
        return "missing"
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("mode:"):
            value = stripped.split(":", 1)[1].strip()
            return value or "unspecified"
    return "unspecified"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a command and capture raw logs for a run."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier.")
    parser.add_argument(
        "--results-root",
        default="results/raw",
        help="Root folder for raw run artifacts. Default: results/raw",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Use '--' before command arguments.",
    )
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided. Usage: run_capture.py --run-id <id> -- <cmd>")

    run_dir = Path(args.results_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    metadata_path = run_dir / "metadata.json"
    metadata_jsonl_path = run_dir / "metadata.jsonl"
    _migrate_legacy_metadata(metadata_path, metadata_jsonl_path)

    started_at = _iso_now()
    start_monotonic = time.monotonic()
    command_str = subprocess.list2cmdline(command)
    stdout_prepend_newline = stdout_path.exists() and stdout_path.stat().st_size > 0
    stderr_prepend_newline = stderr_path.exists() and stderr_path.stat().st_size > 0
    stdout_bytes_before = stdout_path.stat().st_size if stdout_path.exists() else 0
    stderr_bytes_before = stderr_path.stat().st_size if stderr_path.exists() else 0

    with stdout_path.open("ab") as out_file, stderr_path.open("ab") as err_file:
        _append_log_marker(
            out_file,
            timestamp=started_at,
            command_str=command_str,
            prepend_newline=stdout_prepend_newline,
        )
        _append_log_marker(
            err_file,
            timestamp=started_at,
            command_str=command_str,
            prepend_newline=stderr_prepend_newline,
        )

        proc = subprocess.Popen(
            command,
            stdout=out_file,
            stderr=err_file,
            shell=False,
        )
        return_code = proc.wait()
        finished_at = _iso_now()

    duration_sec = round(time.monotonic() - start_monotonic, 6)
    stdout_sha256 = _sha256_file(stdout_path)
    stderr_sha256 = _sha256_file(stderr_path)
    command_sha256 = hashlib.sha256(command_str.encode("utf-8")).hexdigest()

    metadata = {
        "run_id": args.run_id,
        "timestamp_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_sec": duration_sec,
        "duration_ms": int(duration_sec * 1000),
        #"command": command,
        "command_str": command_str,
        "command_sha256": command_sha256,
        "cwd": str(Path.cwd()),
        #"hostname": platform.node() or "unknown",
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "manifest_mode": _read_manifest_mode(run_dir),
        "exit_code": return_code,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "stdout_bytes_before": stdout_bytes_before,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes_before": stderr_bytes_before,
        "stderr_bytes": stderr_path.stat().st_size,
        "stdout_log": str(stdout_path.as_posix()),
        "stderr_log": str(stderr_path.as_posix()),
    }

    with metadata_jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, sort_keys=True))
        handle.write("\n")

    print(f"Captured stdout: {stdout_path}")
    print(f"Captured stderr: {stderr_path}")
    print(f"Appended metadata: {metadata_jsonl_path}")
    print(f"Command exit code: {return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
