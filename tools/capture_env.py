#!/usr/bin/env python3
"""Capture environment metadata for a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional


def _run_text(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _safe_pkg_version(package_name: str) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _safe_module_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _safe_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return []
    try:
        return [str(provider) for provider in ort.get_available_providers()]
    except Exception:
        return []


def _version_from_pinned_freeze(
    lines: list[str],
    package_names: list[str],
) -> Optional[str]:
    package_keys = [name.lower() for name in package_names]
    for line in lines:
        lower = line.lower()
        for key in package_keys:
            prefix = f"{key}=="
            if lower.startswith(prefix):
                return line.split("==", 1)[1].strip()
    return None


def _matching_freeze_lines(lines: list[str], prefixes: list[str]) -> list[str]:
    lowered = [prefix.lower() for prefix in prefixes]
    matches: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in lowered):
            matches.append(line)
    return matches


def _resolve_version(
    metadata_version: Optional[str],
    module_version: Optional[str],
    pinned_freeze_version: Optional[str],
) -> tuple[str, str]:
    if metadata_version:
        return metadata_version, "package_metadata"
    if module_version:
        return module_version, "module_import"
    if pinned_freeze_version:
        return pinned_freeze_version, "pip_freeze_pinned"
    return "not_installed", "none"


def _parse_lemonade_version(raw_output: str) -> Optional[str]:
    for line in raw_output.splitlines():
        match = re.search(
            r"lemonade-server\s+version\s+([0-9A-Za-z][0-9A-Za-z.\-+]*)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

    fallback = re.search(
        r"(?:^|[^0-9])v?([0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.\-]+)?)\b",
        raw_output,
        flags=re.IGNORECASE,
    )
    if fallback:
        return fallback.group(1).strip()
    return None


def _safe_lemonade_server_version() -> str:
    try:
        returncode, stdout, stderr = _run_text(["lemonade-server", "-v"])
    except OSError:
        return "not_installed"

    combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part.strip())
    parsed = _parse_lemonade_version(combined)
    if parsed:
        return parsed
    if returncode != 0:
        return "not_installed"
    return "unknown"


def _as_text(data: dict, pip_freeze: str) -> str:
    lines = [
        f"captured_at={data['captured_at']}",
        f"python_executable={data['python_executable']}",
        f"python_version={data['python_version']}",
        f"platform={data['platform']}",
        f"system={data['system']}",
        f"release={data['release']}",
        f"machine_arch={data['machine_arch']}",
        #f"hostname={data['hostname']}",
        f"onnxruntime_version={data['onnxruntime_version']}",
        f"onnxruntime_version_source={data['onnxruntime_version_source']}",
        f"onnxruntime_genai_version={data['onnxruntime_genai_version']}",
        f"onnxruntime_genai_version_source={data['onnxruntime_genai_version_source']}",
        f"onnxruntime_available_providers={';'.join(data['onnxruntime_available_providers'])}",
        f"lemonade_version={data['lemonade_version']}",
        f"pip_freeze_returncode={data['pip_freeze_returncode']}",
        f"pip_freeze_sha256={data['pip_freeze_sha256']}",
        "",
        "[onnxruntime_packages]",
        "\n".join(data["onnxruntime_packages"]) or "(none)",
        "",
        "[onnxruntime_genai_packages]",
        "\n".join(data["onnxruntime_genai_packages"]) or "(none)",
        "",
        "[pip_freeze]",
        pip_freeze.rstrip(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture Python and package environment snapshot for a run."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier.")
    parser.add_argument(
        "--results-root",
        default="results/raw",
        help="Root folder for raw run artifacts. Default: results/raw",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite env files if they already exist for this run_id.",
    )
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env_json_path = run_dir / "env.json"
    env_txt_path = run_dir / "env.txt"
    if not args.force and (env_json_path.exists() or env_txt_path.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing env snapshot for run_id '{args.run_id}'. "
            "Use --force if overwrite is intentional."
        )

    pip_rc, pip_stdout, pip_stderr = _run_text([sys.executable, "-m", "pip", "freeze"])
    pip_stdout = pip_stdout or ""
    pip_stderr = pip_stderr or ""
    pip_hash = hashlib.sha256(pip_stdout.encode("utf-8")).hexdigest()
    pip_lines = [line for line in pip_stdout.splitlines() if line.strip()]

    onnxruntime_packages = _matching_freeze_lines(
        pip_lines,
        [
            "onnxruntime==",
            "onnxruntime-",
        ],
    )
    onnxruntime_genai_packages = _matching_freeze_lines(
        pip_lines,
        [
            "onnxruntime-genai==",
            "onnxruntime-genai-",
            "onnxruntime_genai",
        ],
    )

    onnxruntime_version, onnxruntime_version_source = _resolve_version(
        metadata_version=_safe_pkg_version("onnxruntime"),
        module_version=_safe_module_version("onnxruntime"),
        pinned_freeze_version=_version_from_pinned_freeze(
            pip_lines,
            ["onnxruntime", "onnxruntime-vitisai"],
        ),
    )
    onnxruntime_genai_version, onnxruntime_genai_version_source = _resolve_version(
        metadata_version=_safe_pkg_version("onnxruntime-genai"),
        module_version=_safe_module_version("onnxruntime_genai"),
        pinned_freeze_version=_version_from_pinned_freeze(
            pip_lines,
            ["onnxruntime-genai", "onnxruntime-genai-directml-ryzenai"],
        ),
    )

    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine_arch": platform.machine(),
        #"hostname": platform.node() or "unknown",
        "onnxruntime_version": onnxruntime_version,
        "onnxruntime_version_source": onnxruntime_version_source,
        "onnxruntime_genai_version": onnxruntime_genai_version,
        "onnxruntime_genai_version_source": onnxruntime_genai_version_source,
        "onnxruntime_available_providers": _safe_ort_providers(),
        "onnxruntime_packages": onnxruntime_packages,
        "onnxruntime_genai_packages": onnxruntime_genai_packages,
        "lemonade_version": _safe_lemonade_server_version(),
        "pip_freeze_returncode": pip_rc,
        "pip_freeze_sha256": pip_hash,
        "pip_freeze": pip_lines,
    }
    if pip_rc != 0:
        payload["pip_freeze_stderr"] = pip_stderr.strip()

    env_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    #env_txt_path.write_text(_as_text(payload, pip_stdout), encoding="utf-8")

    print(f"Saved environment snapshot to: {env_json_path}")
    #print(f"Saved text snapshot to: {env_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
