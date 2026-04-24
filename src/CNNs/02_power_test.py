#!/usr/bin/env python3
"""Run one CNN power-test attempt with a single telemetry source."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
UTILS_SCRIPT = SCRIPT_PATH.with_name("utils.py")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "raw"
DEFAULT_SHARED_VAIP_CACHE_DIR = DEFAULT_RESULTS_ROOT / "cnn_power_test"
DEFAULT_IDLE_SECONDS = 30.0
DEFAULT_COOLDOWN_SECONDS = 10.0
DEFAULT_XRT_INTERVAL_SEC = 0.2
DEFAULT_UPROF_INTERVAL_MS = 200
DEFAULT_UPROF_EVENT = "socket=0,power"
UPROF_COUNTER_ERROR = "there is no counters avialable"
XRT_WINDOWS_PATH = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
UPROF_WINDOWS_PATH = Path(r"C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe")


@dataclass(frozen=True)
class PowerTestConfig:
    run_id: str
    case_name: str
    attempt_index: int
    telemetry_tool: str
    model_path: Path
    device: str
    disable_fallback: bool
    input_shape: str
    batch: int
    warmup: int
    repeat: int
    idle_seconds: float
    cooldown_seconds: float
    python_exe: Path
    smoke_script: Path
    results_root: Path
    run_dir: Path
    attempt_dir: Path
    shared_vaip_cache_dir: Path | None
    vaip_cache_key: str | None
    uprof_exe: Path | None
    uprof_event: str
    uprof_interval_ms: int
    xrt_smi_exe: Path | None
    xrt_interval_sec: float


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    returncode: int
    duration_sec: float
    stdout: str
    stderr: str


class XrtPoller(threading.Thread):
    def __init__(
        self,
        *,
        xrt_smi_exe: Path,
        attempt_dir: Path,
        interval_sec: float,
    ) -> None:
        super().__init__(daemon=True)
        self._xrt_smi_exe = xrt_smi_exe
        self._attempt_dir = attempt_dir
        self._interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._sample_path = attempt_dir / "xrt_samples.jsonl"
        self._errors: list[str] = []
        self.sample_count = 0

    @property
    def sample_path(self) -> Path:
        return self._sample_path

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            start_time = time.time()
            self._capture_sample()
            elapsed = time.time() - start_time
            wait_time = max(0.0, self._interval_sec - elapsed)
            self._stop_event.wait(wait_time)

    def _capture_sample(self) -> None:
        sample_index = self.sample_count + 1
        temp_path = self._attempt_dir / f"xrt_sample_{sample_index:04d}.json"
        command = [
            str(self._xrt_smi_exe),
            "examine",
            "--report",
            "platform",
            "--format",
            "JSON",
            "--output",
            str(temp_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        payload: dict[str, Any] = {
            "captured_at_utc": iso_utc_now(),
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        if temp_path.exists():
            try:
                payload["report"] = json.loads(temp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload["parse_error"] = f"invalid_json:{exc}"
            finally:
                temp_path.unlink(missing_ok=True)
        elif completed.returncode == 0:
            payload["parse_error"] = "missing_output_file"

        with self._sample_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

        self.sample_count += 1
        if completed.returncode != 0:
            self._errors.append(
                f"xrt-smi sample {sample_index} failed with exit code {completed.returncode}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one CNN power-test attempt with one telemetry tool."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier under results/raw.")
    parser.add_argument(
        "--case-name",
        default=None,
        help="Stable case label. Defaults to a name derived from model precision and device.",
    )
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help="1-based repetition index for the case/tool combination.",
    )
    parser.add_argument(
        "--telemetry-tool",
        required=True,
        choices=("uprof", "xrt"),
        help="Telemetry tool used for this attempt.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the ONNX model.")
    parser.add_argument(
        "--device",
        required=True,
        choices=("cpu", "npu", "igpu"),
        help="Execution target device.",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable ONNX Runtime execution-provider fallback.",
    )
    parser.add_argument("--input-shape", default="3x224x224", help="Input tensor shape.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=20000, help="Timed iterations.")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=DEFAULT_IDLE_SECONDS,
        help="Idle capture duration before the measured workload for non-xrt tools.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Cooldown gap between idle capture and workload capture for non-xrt tools.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch 01_cnn_smoke_test.py.",
    )
    parser.add_argument(
        "--smoke-script",
        default=str(REPO_ROOT / "src" / "CNNs" / "01_cnn_smoke_test.py"),
        help="Path to the smoke-test workload script.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root folder for raw results.",
    )
    parser.add_argument(
        "--shared-vaip-cache-dir",
        default=str(DEFAULT_SHARED_VAIP_CACHE_DIR),
        help="Shared VAIP cache directory reused for NPU runs.",
    )
    parser.add_argument(
        "--vaip-cache-key",
        default=None,
        help="NPU cache key inside --shared-vaip-cache-dir.",
    )
    parser.add_argument(
        "--uprof-exe",
        default=None,
        help="Path to AMDuProfCLI.exe. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--uprof-event",
        default=DEFAULT_UPROF_EVENT,
        help="AMDuProfCLI timechart event selector.",
    )
    parser.add_argument(
        "--uprof-interval-ms",
        type=int,
        default=DEFAULT_UPROF_INTERVAL_MS,
        help="AMDuProfCLI timechart sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--xrt-smi-exe",
        default=None,
        help="Path to xrt-smi.exe. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--xrt-interval-sec",
        type=float,
        default=DEFAULT_XRT_INTERVAL_SEC,
        help="Polling interval for xrt-smi sampling.",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_optional_tool_path(raw_path: str | None, fallback: Path) -> Path | None:
    if raw_path:
        return Path(raw_path).resolve()
    discovered = shutil.which(fallback.name)
    if discovered:
        return Path(discovered).resolve()
    if fallback.exists():
        return fallback.resolve()
    return None


def derive_case_name(model_path: Path, device: str) -> str:
    stem = model_path.stem.lower()
    if "a8w8" in stem or "int8" in stem:
        precision = "int8"
    elif "fp16" in stem:
        precision = "fp16"
    elif "bf16" in stem:
        precision = "bf16"
    else:
        precision = "fp32"
    return f"{precision}_{device}"


def build_config(args: argparse.Namespace) -> PowerTestConfig:
    if args.attempt_index <= 0:
        raise SystemExit("--attempt-index must be positive.")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be zero or positive.")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive.")
    if args.idle_seconds < 0:
        raise SystemExit("--idle-seconds must be zero or positive.")
    if args.cooldown_seconds < 0:
        raise SystemExit("--cooldown-seconds must be zero or positive.")
    if args.xrt_interval_sec <= 0:
        raise SystemExit("--xrt-interval-sec must be positive.")
    if args.uprof_interval_ms <= 0:
        raise SystemExit("--uprof-interval-ms must be positive.")

    model_path = resolve_repo_path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    smoke_script = resolve_repo_path(args.smoke_script)
    if not smoke_script.exists():
        raise SystemExit(f"Smoke-test script not found: {smoke_script}")

    python_exe = Path(args.python_exe).resolve()
    if not python_exe.exists():
        discovered_python = shutil.which(args.python_exe)
        if not discovered_python:
            raise SystemExit(f"Python executable not found: {args.python_exe}")
        python_exe = Path(discovered_python).resolve()

    results_root = resolve_repo_path(args.results_root)
    run_dir = results_root / args.run_id
    case_name = args.case_name or derive_case_name(model_path, args.device)
    attempt_dir = run_dir / "power" / args.telemetry_tool / case_name / f"attempt_{args.attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    shared_vaip_cache_dir: Path | None = None
    if args.device == "npu":
        shared_vaip_cache_dir = resolve_repo_path(args.shared_vaip_cache_dir)
        if not args.vaip_cache_key:
            raise SystemExit("--vaip-cache-key is required for NPU runs.")
        if not shared_vaip_cache_dir.exists():
            raise SystemExit(
                f"Shared VAIP cache directory does not exist: {shared_vaip_cache_dir}"
            )

    uprof_exe = resolve_optional_tool_path(args.uprof_exe, UPROF_WINDOWS_PATH)
    xrt_smi_exe = resolve_optional_tool_path(args.xrt_smi_exe, XRT_WINDOWS_PATH)
    if args.telemetry_tool == "xrt" and args.device != "npu":
        raise SystemExit("xrt telemetry is supported only for NPU runs.")

    return PowerTestConfig(
        run_id=args.run_id,
        case_name=case_name,
        attempt_index=args.attempt_index,
        telemetry_tool=args.telemetry_tool,
        model_path=model_path,
        device=args.device,
        disable_fallback=args.disable_fallback,
        input_shape=args.input_shape,
        batch=args.batch,
        warmup=args.warmup,
        repeat=args.repeat,
        idle_seconds=args.idle_seconds,
        cooldown_seconds=args.cooldown_seconds,
        python_exe=python_exe,
        smoke_script=smoke_script,
        results_root=results_root,
        run_dir=run_dir,
        attempt_dir=attempt_dir,
        shared_vaip_cache_dir=shared_vaip_cache_dir,
        vaip_cache_key=args.vaip_cache_key,
        uprof_exe=uprof_exe,
        uprof_event=args.uprof_event,
        uprof_interval_ms=args.uprof_interval_ms,
        xrt_smi_exe=xrt_smi_exe,
        xrt_interval_sec=args.xrt_interval_sec,
    )


def iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return path_for_output(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def print_json_line(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix}=" + json.dumps(json_ready(payload), sort_keys=True))


def path_for_output(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path.resolve())


def build_workload_command(config: PowerTestConfig) -> list[str]:
    command = [
        str(config.python_exe),
        str(config.smoke_script),
        "--model-path",
        path_for_command(config.model_path),
        "--device",
        config.device,
        "--input-shape",
        config.input_shape,
        "--batch",
        str(config.batch),
        "--warmup",
        str(config.warmup),
        "--repeat",
        str(config.repeat),
    ]
    if config.disable_fallback:
        command.append("--disable-fallback")
    if config.device == "npu":
        if config.shared_vaip_cache_dir is None or not config.vaip_cache_key:
            raise RuntimeError("NPU runs require a shared VAIP cache directory and key.")
        command.extend(
            [
                "--vaip-cache-dir",
                path_for_command(config.shared_vaip_cache_dir),
                "--vaip-cache-key",
                config.vaip_cache_key,
            ]
        )
    return command


def build_idle_command(config: PowerTestConfig) -> list[str]:
    if not UTILS_SCRIPT.exists():
        raise RuntimeError(f"Utility script not found: {UTILS_SCRIPT}")
    return [
        str(config.python_exe),
        str(UTILS_SCRIPT),
        "sleep",
    ]


def path_for_command(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def execute_command(
    command: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> ProcessResult:
    start_time = time.perf_counter()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    return ProcessResult(
        command=command,
        returncode=completed.returncode,
        duration_sec=round(time.perf_counter() - start_time, 6),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def sleep_with_status(seconds: float, label: str) -> None:
    if seconds <= 0:
        return
    print(f"{label}={json.dumps({'seconds': seconds})}")
    time.sleep(seconds)


def emit_process_output(result: ProcessResult) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def summarize_process(result: ProcessResult) -> dict[str, Any]:
    return {
        "command": result.command,
        "duration_sec": result.duration_sec,
        "exit_code": result.returncode,
    }


def summarize_numeric_series(values: list[float]) -> dict[str, Any]:
    ordered = list(values)
    count = len(ordered)
    if count == 0:
        return {"count": 0}
    mean_value = sum(ordered) / count
    minimum = min(ordered)
    maximum = max(ordered)
    variance = 0.0 if count == 1 else sum((value - mean_value) ** 2 for value in ordered) / count
    return {
        "count": count,
        "first": round(ordered[0], 4),
        "last": round(ordered[-1], 4),
        "min": round(minimum, 4),
        "max": round(maximum, 4),
        "mean": round(mean_value, 4),
        "std": round(variance ** 0.5, 4),
    }


def try_parse_float(raw_value: str) -> float | None:
    text = raw_value.strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    if cleaned.upper() in {"N/A", "NA", "NAN", "-", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_csv_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")
    if not sample.strip():
        return [], ","
    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [{key or "": value or "" for key, value in row.items()} for row in reader]
    return rows, delimiter


def select_numeric_columns(rows: list[dict[str, str]], hints: tuple[str, ...] | None = None) -> dict[str, list[float]]:
    values_by_column: dict[str, list[float]] = {}
    for row in rows:
        for key, raw_value in row.items():
            if not key:
                continue
            parsed = try_parse_float(raw_value)
            if parsed is None:
                continue
            values_by_column.setdefault(key, []).append(parsed)

    if not hints:
        return values_by_column

    preferred: dict[str, list[float]] = {}
    for key, values in values_by_column.items():
        normalized = key.lower()
        if any(hint in normalized for hint in hints):
            preferred[key] = values
    if preferred:
        return preferred
    return values_by_column


def summarize_csv_file(path: Path, hints: tuple[str, ...] | None = None) -> dict[str, Any]:
    rows, delimiter = parse_csv_rows(path)
    numeric_columns = select_numeric_columns(rows, hints=hints)
    sorted_items = sorted(numeric_columns.items(), key=lambda item: item[0].lower())
    summary: dict[str, Any] = {
        "delimiter": delimiter,
        "file": path_for_output(path),
        "row_count": len(rows),
        "numeric_column_count": len(sorted_items),
    }
    if rows:
        header = [column for column in rows[0].keys() if column]
        summary["header"] = header
    if sorted_items:
        summary["numeric_columns"] = {
            key: summarize_numeric_series(values) for key, values in sorted_items[:20]
        }
    return summary


def list_existing_paths(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*")}


def list_new_paths(root: Path, before: set[Path]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.resolve() not in before],
        key=lambda path: str(path),
    )


def collect_new_csv_summaries(
    root: Path,
    before: set[Path],
    *,
    hints: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    new_paths = list_new_paths(root, before)
    csv_paths = [path for path in new_paths if path.is_file() and path.suffix.lower() == ".csv"]
    summaries = [summarize_csv_file(path, hints=hints) for path in csv_paths]
    artifacts = [
        path_for_output(path)
        for path in new_paths
        if path.is_file()
    ]
    return summaries, artifacts


def ensure_tool_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise RuntimeError(f"{label} executable not found.")
    if not path.exists():
        raise RuntimeError(f"{label} executable not found: {path}")
    return path


def run_uprof(
    config: PowerTestConfig,
    workload_command: list[str],
) -> tuple[ProcessResult, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    uprof_exe = ensure_tool_path(config.uprof_exe, "AMD uProf")
    preflight = execute_command([str(uprof_exe), "timechart", "--list"])
    preflight_text = f"{preflight.stdout}\n{preflight.stderr}".lower()
    if preflight.returncode != 0 or UPROF_COUNTER_ERROR in preflight_text:
        raise RuntimeError(
            "AMD uProf timechart did not report usable counters. "
            "Run the terminal as Administrator and confirm AMDuProfCLI timechart can access socket power counters."
        )

    idle_root = config.attempt_dir / "idle"
    workload_root = config.attempt_dir / "workload"
    idle_before = list_existing_paths(idle_root)
    workload_before = list_existing_paths(workload_root)

    idle_output_prefix = config.attempt_dir / "idle" / "uprof"
    idle_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    idle_command = [
        str(uprof_exe),
        "timechart",
        "--event",
        config.uprof_event,
        "--interval",
        str(config.uprof_interval_ms),
        "-o",
        str(idle_output_prefix),
        *build_idle_command(config),
    ]
    idle_result = execute_command(
        idle_command,
        extra_env={"CNN_SLEEP_SECONDS": str(config.idle_seconds)},
    )

    sleep_with_status(config.cooldown_seconds, "power_cooldown")

    workload_output_prefix = config.attempt_dir / "workload" / "uprof"
    workload_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    workload_capture_command = [
        str(uprof_exe),
        "timechart",
        "--event",
        config.uprof_event,
        "--interval",
        str(config.uprof_interval_ms),
        "-o",
        str(workload_output_prefix),
        *workload_command,
    ]
    result = execute_command(workload_capture_command)

    idle_csv_summaries, idle_artifact_files = collect_new_csv_summaries(
        idle_root,
        idle_before,
        hints=("power",),
    )
    workload_csv_summaries, workload_artifact_files = collect_new_csv_summaries(
        workload_root,
        workload_before,
        hints=("power",),
    )
    telemetry_summary = {
        "event": config.uprof_event,
        "interval_ms": config.uprof_interval_ms,
        "idle": {
            "capture": summarize_process(idle_result),
            "csv_summaries": idle_csv_summaries,
        },
        "workload": {
            "capture": summarize_process(result),
            "csv_summaries": workload_csv_summaries,
        },
        "cooldown_seconds": config.cooldown_seconds,
        "preflight": summarize_process(preflight),
    }
    artifacts = {
        "artifact_dir": config.attempt_dir,
        "files": idle_artifact_files + workload_artifact_files,
    }
    return result, telemetry_summary, artifacts, summarize_process(idle_result)


def summarize_xrt_samples(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sample_count": 0}

    sample_count = 0
    power_values: list[float] = []
    power_mode_values: list[str] = []
    device_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            sample_count += 1
            payload = json.loads(line)
            report = payload.get("report")
            if not isinstance(report, dict):
                continue
            for device in report.get("devices", []):
                if isinstance(device, dict):
                    device_id = device.get("device_id")
                    if isinstance(device_id, str):
                        device_ids.add(device_id)
                    platforms = device.get("platforms", [])
                    if not isinstance(platforms, list):
                        continue
                    for platform in platforms:
                        if not isinstance(platform, dict):
                            continue
                        status = platform.get("status", {})
                        if isinstance(status, dict):
                            power_mode = status.get("power_mode")
                            if isinstance(power_mode, str) and power_mode:
                                power_mode_values.append(power_mode)
                        electrical = platform.get("electrical", {})
                        if isinstance(electrical, dict):
                            watts = electrical.get("power_consumption_watts")
                            if isinstance(watts, str):
                                parsed = try_parse_float(watts)
                                if parsed is not None:
                                    power_values.append(parsed)

    summary: dict[str, Any] = {
        "device_ids": sorted(device_ids),
        "sample_count": sample_count,
        "sample_file": path_for_output(path),
    }
    if power_values:
        summary["power_consumption_watts"] = summarize_numeric_series(power_values)
    if power_mode_values:
        summary["power_modes_observed"] = sorted(set(power_mode_values))
    return summary


def run_xrt(
    config: PowerTestConfig,
    workload_command: list[str],
) -> tuple[ProcessResult, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    xrt_smi_exe = ensure_tool_path(config.xrt_smi_exe, "xrt-smi")
    poller = XrtPoller(
        xrt_smi_exe=xrt_smi_exe,
        attempt_dir=config.attempt_dir,
        interval_sec=config.xrt_interval_sec,
    )
    poller.start()
    try:
        result = execute_command(workload_command)
    finally:
        poller.stop()
        poller.join(timeout=max(5.0, config.xrt_interval_sec * 2.0))

    telemetry_summary = summarize_xrt_samples(poller.sample_path)
    if poller.errors:
        telemetry_summary["poller_errors"] = poller.errors
    artifacts = {
        "artifact_dir": config.attempt_dir,
        "xrt_samples": poller.sample_path,
    }
    return result, telemetry_summary, artifacts, None


def build_case_summary(config: PowerTestConfig, workload_command: list[str]) -> dict[str, Any]:
    return {
        "attempt_dir": config.attempt_dir,
        "attempt_index": config.attempt_index,
        "batch": config.batch,
        "case_name": config.case_name,
        "device": config.device,
        "disable_fallback": config.disable_fallback,
        "idle_seconds": config.idle_seconds,
        "input_shape": config.input_shape,
        "model_path": config.model_path,
        "repeat": config.repeat,
        "run_id": config.run_id,
        "shared_vaip_cache_dir": config.shared_vaip_cache_dir,
        "cooldown_seconds": config.cooldown_seconds,
        "telemetry_tool": config.telemetry_tool,
        "uprof_interval_ms": config.uprof_interval_ms,
        "vaip_cache_key": config.vaip_cache_key,
        "warmup": config.warmup,
        "workload_command": workload_command,
        "xrt_interval_sec": config.xrt_interval_sec,
    }


def main() -> int:
    try:
        config = build_config(parse_args())
        workload_command = build_workload_command(config)

        print_json_line("power_case", build_case_summary(config, workload_command))

        if config.telemetry_tool == "uprof":
            result, telemetry_summary, artifacts, idle_summary = run_uprof(config, workload_command)
        else:
            result, telemetry_summary, artifacts, idle_summary = run_xrt(config, workload_command)

        emit_process_output(result)
        if idle_summary is not None:
            print_json_line("power_idle", idle_summary)
        print_json_line("power_workload", summarize_process(result))
        print_json_line("telemetry_summary", telemetry_summary)
        print_json_line("artifacts", artifacts)
        return result.returncode
    except RuntimeError as exc:
        print(f"power_error={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
