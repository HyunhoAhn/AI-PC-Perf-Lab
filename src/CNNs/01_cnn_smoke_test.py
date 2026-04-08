#!/usr/bin/env python3
"""CNN smoke test runner for ONNX Runtime providers."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


DEVICE_TO_PROVIDER = {
    "npu": "VitisAIExecutionProvider",
    "igpu": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

ORT_TO_NUMPY_DTYPE = {
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


@dataclass(frozen=True)
class RunConfig:
    model_path: Path
    #precision: str
    device: str
    provider: str
    disable_fallback: bool
    input_shape: list[int]
    batch: int
    repeat: int
    warmup: int
    profile_out: Path | None
    requires_custom_ops: bool
    vaip_cache_dir: Path | None
    vaip_cache_key: str | None
    clear_vaip_cache: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CNN smoke test with ONNX Runtime.")
    parser.add_argument("--model-path", required=True, help="Path to the ONNX model.")
    parser.add_argument(
        "--device",
        required=True,
        choices=tuple(DEVICE_TO_PROVIDER),
        help="Execution target device.",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable ONNX Runtime run-time execution-provider fallback.",
    )
    parser.add_argument(
        "--input-shape",
        default="3x224x224",
        help="Input tensor shape excluding batch. Format: 3x224x224",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size override.")
    parser.add_argument("--repeat", type=int, default=30, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument(
        "--profile-out",
        default=None,
        help="Write ONNX Runtime profiling JSON to this path.",
    )
    parser.add_argument(
        "--vaip-cache-dir",
        default=None,
        help="NPU only. Directory used for Vitis AI EP cache.",
    )
    parser.add_argument(
        "--vaip-cache-key",
        default=None,
        help="NPU only. Cache subfolder key inside --vaip-cache-dir.",
    )
    parser.add_argument(
        "--clear-vaip-cache",
        action="store_true",
        help="NPU only. Clear only the requested VAIP cache-key subfolder before creating the session.",
    )
    '''
    parser.add_argument(
        "--precision",
        required=True,
        choices=("fp32", "fp16", "int8", "bf16"),
        help="Model precision label for reporting.",
    )
    '''
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be zero or positive.")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    vaip_options_used = any(
        (
            args.vaip_cache_dir,
            args.vaip_cache_key,
            args.clear_vaip_cache,
        )
    )
    if args.device != "npu" and vaip_options_used:
        raise SystemExit(
            "--vaip-cache-dir, --vaip-cache-key, and --clear-vaip-cache "
            "can only be used with --device npu."
        )

    vaip_cache_dir = Path(args.vaip_cache_dir) if args.vaip_cache_dir else None
    if args.clear_vaip_cache and vaip_cache_dir is None:
        raise SystemExit("--clear-vaip-cache requires --vaip-cache-dir.")
    if args.clear_vaip_cache and not args.vaip_cache_key:
        raise SystemExit("--clear-vaip-cache requires --vaip-cache-key.")

    return RunConfig(
        model_path=model_path,
        #precision=args.precision,
        device=args.device,
        provider=resolve_provider(args.device),
        disable_fallback=args.disable_fallback,
        input_shape=parse_shape(args.input_shape, args.batch),
        batch=args.batch,
        repeat=args.repeat,
        warmup=args.warmup,
        profile_out=Path(args.profile_out) if args.profile_out else None,
        requires_custom_ops=model_requires_custom_ops(model_path),
        vaip_cache_dir=vaip_cache_dir,
        vaip_cache_key=args.vaip_cache_key,
        clear_vaip_cache=args.clear_vaip_cache,
    )


def parse_shape(shape_text: str, batch: int) -> list[int]:
    parts = [part.strip() for part in shape_text.lower().split("x") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid input shape: {shape_text!r}")

    try:
        shape = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid input shape: {shape_text!r}") from exc

    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Input shape dimensions must be positive: {shape_text!r}")
    if batch <= 0:
        raise ValueError("Batch size must be positive.")

    if len(shape) == 3:
        return [batch, *shape]
    raise ValueError(
        f"Input shape must be CHW (for example 3x224x224) with optional batch dimension. Got: {shape_text!r}"
    )


def ensure_parent_path(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directory_path(path: Path | None) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def print_json_line(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix}=" + json.dumps(payload, sort_keys=True))

def clear_vaip_cache_if_requested(config: RunConfig) -> None:
    if not config.clear_vaip_cache:
        return

    if config.vaip_cache_dir is None or not config.vaip_cache_key:
        return

    cache_path = config.vaip_cache_dir / config.vaip_cache_key
    if cache_path.exists():
        shutil.rmtree(cache_path)


def build_provider_options(config: RunConfig) -> list[dict[str, str]] | None:
    if config.device != "npu":
        return None

    provider_options: dict[str, str] = {}
    if config.vaip_cache_dir is not None:
        provider_options["cache_dir"] = str(config.vaip_cache_dir.resolve())
    if config.vaip_cache_key:
        provider_options["cache_key"] = config.vaip_cache_key

    if not provider_options:
        return None
    return [provider_options]

def model_requires_custom_ops(model_path: Path) -> bool:
    model = onnx.load(str(model_path), load_external_data=False)
    return any(node.domain == "com.amd.quark" for node in model.graph.node)


def resolve_provider(device: str) -> str:
    provider = DEVICE_TO_PROVIDER[device]
    available = ort.get_available_providers()
    if provider not in available:
        available_text = ", ".join(available) if available else "<none>"
        raise RuntimeError(
            f"Requested provider {provider} for device={device!r} is not available. "
            f"Available providers: {available_text}"
        )
    return provider


def register_custom_ops_if_needed(
    sess_options: ort.SessionOptions,
    requires_custom_ops: bool,
) -> bool:
    if not requires_custom_ops:
        return False

    try:
        from quark.onnx import get_library_path
    except ImportError:
        try:
            from quark.onnx.operators.custom_ops import get_library_path
        except ImportError as exc:
            raise RuntimeError(
                "Quark custom ops are required for NPU runs but could not be imported."
            ) from exc

    sess_options.register_custom_ops_library(get_library_path())
    return True


def ort_type_to_numpy_dtype(ort_type: str) -> np.dtype[Any]:
    try:
        return np.dtype(ORT_TO_NUMPY_DTYPE[ort_type])
    except KeyError as exc:
        raise RuntimeError(f"Unsupported ONNX Runtime tensor type: {ort_type}") from exc


def normalize_metadata_shape(raw_shape: list[Any], batch: int) -> list[int]:
    '''Given a raw shape from ONNX Runtime input metadata, produce a fully specified shape for input generation.
    The raw shape may contain symbolic dimensions (represented as strings) or dynamic dimensions (represented as -1). The first dimension is assumed to be batch size if it is not a positive integer.
    '''
    normalized: list[int] = []
    for index, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        elif index == 0:
            normalized.append(batch)
        else:
            normalized.append(1)
    return normalized


def build_input_tensor(
    *,
    dtype: np.dtype[Any],
    shape: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.int8).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 10, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(size=shape).astype(dtype)
    raise RuntimeError(f"Unsupported numpy dtype for input generation: {dtype}")


def build_inputs(
    session: ort.InferenceSession,
    first_input_shape: list[int],
    batch: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    feeds: dict[str, np.ndarray] = {}
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs.")

    for index, input_meta in enumerate(inputs):
        dtype = ort_type_to_numpy_dtype(input_meta.type)
        if index == 0:
            shape = first_input_shape
        else:
            shape = normalize_metadata_shape(list(input_meta.shape), batch)
        feeds[input_meta.name] = build_input_tensor(dtype=dtype, shape=shape, rng=rng)
    return feeds


def create_session(config: RunConfig) -> tuple[ort.InferenceSession, bool]:
    session_options = ort.SessionOptions()

    if config.disable_fallback:
        session_options.add_session_config_entry(
            "session.disable_cpu_ep_fallback",
            "1",
        )

    if config.profile_out is not None:
        session_options.enable_profiling = True

    custom_ops_registered = register_custom_ops_if_needed(
        session_options,
        config.requires_custom_ops,
    )
    if config.device == "npu" and config.vaip_cache_dir is not None:
        ensure_directory_path(config.vaip_cache_dir)
    clear_vaip_cache_if_requested(config)

    session = ort.InferenceSession(
        str(config.model_path),
        sess_options=session_options,
        providers=[config.provider],
        provider_options=build_provider_options(config),
    )

    if config.disable_fallback:
        try:
            session.disable_fallback()
        except AttributeError:
            pass

    return session, custom_ops_registered


def summarize_latencies(latencies_ms: list[float], batch: int) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    throughput = 0.0 if mean_ms == 0.0 else float((batch * 1000.0) / mean_ms)
    return {
        "count": int(values.size),
        "min_ms": round(float(values.min()), 4),
        "mean_ms": round(mean_ms, 4),
        "max_ms": round(float(values.max()), 4),
        "std_ms": round(float(values.std(ddof=0)), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p90_ms": round(float(np.percentile(values, 90)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "throughput_items_per_sec": round(throughput, 4),
    }


def build_profile_destination(profile_out: Path, generated_profile: Path) -> Path:
    if profile_out.suffix.lower() == ".json":
        return profile_out

    return profile_out / generated_profile.name


def write_profile_if_enabled(
    session: ort.InferenceSession,
    profile_out: Path | None,
    config: RunConfig,
) -> str | None:
    if profile_out is None:
        return None

    generated_profile = session.end_profiling()
    source = Path(generated_profile)
    destination = build_profile_destination(profile_out, source)
    ensure_parent_path(destination)
    if source.resolve() != destination.resolve():
        shutil.move(str(source), str(destination))
    return str(destination)


def build_config_summary(config: RunConfig, custom_ops_registered: bool) -> dict[str, Any]:
    return {
        "available_providers": ort.get_available_providers(),
        "batch": config.batch,
        "custom_ops_registered": custom_ops_registered,
        "device": config.device,
        "disable_fallback": config.disable_fallback,
        "input_shape": config.input_shape,
        "model_path": str(config.model_path),
        #"precision": config.precision,
        "profile_out": str(config.profile_out) if config.profile_out else None,
        "provider": config.provider,
        "repeat": config.repeat,
        "requires_custom_ops": config.requires_custom_ops,
        "vaip_cache_dir": str(config.vaip_cache_dir) if config.vaip_cache_dir else None,
        "vaip_cache_key": config.vaip_cache_key,
        "clear_vaip_cache": config.clear_vaip_cache,
        "warmup": config.warmup,
    }


def build_session_summary(
    session: ort.InferenceSession,
    output_names: list[str],
) -> dict[str, Any]:
    return {
        "input_names": [item.name for item in session.get_inputs()],
        "input_types": {item.name: item.type for item in session.get_inputs()},
        "output_names": output_names,
        "output_types": {item.name: item.type for item in session.get_outputs()},
    }


def run_inference_loop(
    session: ort.InferenceSession,
    output_names: list[str],
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    for _ in range(warmup):
        session.run(output_names, feeds)

    latencies_ms: list[float] = []
    for _ in range(repeat):
        start_ns = time.perf_counter_ns()
        session.run(output_names, feeds)
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
    return latencies_ms


def collect_artifacts(
    session: ort.InferenceSession,
    config: RunConfig,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    profile_out = write_profile_if_enabled(session, config.profile_out, config)
    if profile_out is not None:
        artifacts["profile_out"] = profile_out

    return artifacts


def main() -> int:
    config = build_config(parse_args())
    session, custom_ops_registered = create_session(config)
    output_names = [output.name for output in session.get_outputs()]
    feeds = build_inputs(session, config.input_shape, config.batch)

    print_json_line("config", build_config_summary(config, custom_ops_registered))
    print_json_line("session", build_session_summary(session, output_names))

    latencies_ms = run_inference_loop(
        session,
        output_names,
        feeds,
        warmup=config.warmup,
        repeat=config.repeat,
    )
    print_json_line("timing", summarize_latencies(latencies_ms, config.batch))

    artifacts = collect_artifacts(session, config)
    if artifacts:
        print_json_line("artifacts", artifacts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
