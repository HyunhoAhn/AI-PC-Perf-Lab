#!/usr/bin/env python3
"""Compare two CPU ONNX models with random image-like inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two CPU ONNX models on the same random inputs and compare outputs."
    )
    parser.add_argument("--model-a", required=True, help="Path to the first ONNX model.")
    parser.add_argument("--model-b", required=True, help="Path to the second ONNX model.")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional directory of real images. If omitted, random inputs are used.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of inputs to compare. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible inputs. Default: 0.",
    )
    return parser.parse_args()


def create_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def infer_input_shape(session: ort.InferenceSession) -> list[int]:
    input_meta = session.get_inputs()[0]
    raw_shape = list(input_meta.shape)
    if len(raw_shape) != 4:
        raise SystemExit(f"Expected 4D input, got {raw_shape!r}")

    normalized: list[int] = []
    for index, dim in enumerate(raw_shape):
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        elif index == 0:
            normalized.append(1)
        else:
            normalized.append(224 if index in (2, 3) else 3)

    if normalized[0] != 1:
        normalized[0] = 1

    chw = normalized[1:] == [3, 224, 224]
    hwc = normalized[1:] == [224, 224, 3]
    if not (chw or hwc):
        raise SystemExit(
            f"Unsupported input shape {normalized!r}. Expected batch-1 CHW or HWC image input."
        )
    return normalized


def build_random_input(shape: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(np.float32)


def collect_image_paths(image_dir: Path, limit: int) -> list[Path]:
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"--image-dir must point to a directory: {image_dir}")

    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not image_paths:
        raise SystemExit(f"No supported image files found under: {image_dir}")
    return image_paths[:limit]


def resize_shorter_side(image: Image.Image, target_short_side: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise SystemExit(f"Invalid image size: {image.size!r}")

    if width < height:
        new_width = target_short_side
        new_height = int(round(height * target_short_side / width))
    else:
        new_height = target_short_side
        new_width = int(round(width * target_short_side / height))

    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max((width - crop_size) // 2, 0)
    top = max((height - crop_size) // 2, 0)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def build_image_input(shape: list[int], image_path: Path) -> np.ndarray:
    chw = shape[1:] == [3, 224, 224]
    hwc = shape[1:] == [224, 224, 3]
    if not (chw or hwc):
        raise SystemExit(f"Unsupported image input shape: {shape!r}")

    image = Image.open(image_path).convert("RGB")
    image = resize_shorter_side(image, target_short_side=256)
    image = center_crop(image, crop_size=224)

    array = np.asarray(image, dtype=np.float32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    array = (array - mean) / std

    if chw:
        array = np.transpose(array, (2, 0, 1))

    return np.expand_dims(array, axis=0).astype(np.float32)


def run_model(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    if len(outputs) != 1:
        raise SystemExit(f"Expected 1 output tensor, got {len(outputs)}")
    return np.asarray(outputs[0], dtype=np.float64)


def topk_indices(values: np.ndarray, k: int) -> list[int]:
    if values.size < k:
        raise SystemExit(f"Output size {values.size} is smaller than top-{k}.")
    return np.argsort(values)[-k:][::-1].astype(int).tolist()


def summarize_diff(
    output_a: np.ndarray, output_b: np.ndarray
) -> dict[str, float | bool | int | list[int]]:
    if output_a.shape != output_b.shape:
        raise SystemExit(f"Output shape mismatch: {output_a.shape!r} vs {output_b.shape!r}")

    flat_a = output_a.reshape(-1)
    flat_b = output_b.reshape(-1)
    diff = flat_a - flat_b
    abs_diff = np.abs(diff)

    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    if norm_a == 0.0 and norm_b == 0.0:
        cosine = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

    argmax_a = int(np.argmax(flat_a))
    argmax_b = int(np.argmax(flat_b))
    top5_a = topk_indices(flat_a, k=5)
    top5_b = topk_indices(flat_b, k=5)
    top5_overlap = len(set(top5_a) & set(top5_b))

    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine_similarity": cosine,
        "argmax_a": argmax_a,
        "argmax_b": argmax_b,
        "argmax_match": argmax_a == argmax_b,
        "top5_a": top5_a,
        "top5_b": top5_b,
        "top5_exact_match": top5_a == top5_b,
        "top5_overlap": top5_overlap,
    }


def main() -> int:
    args = parse_args()
    model_a = Path(args.model_a)
    model_b = Path(args.model_b)
    if not model_a.exists():
        raise SystemExit(f"Model file not found: {model_a}")
    if not model_b.exists():
        raise SystemExit(f"Model file not found: {model_b}")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")

    session_a = create_session(model_a)
    session_b = create_session(model_b)

    shape_a = infer_input_shape(session_a)
    shape_b = infer_input_shape(session_b)
    if shape_a != shape_b:
        raise SystemExit(f"Input shape mismatch: {shape_a!r} vs {shape_b!r}")

    image_paths: list[Path] = []
    if args.image_dir:
        image_paths = collect_image_paths(Path(args.image_dir), limit=args.samples)

    print(f"model_a: {model_a}")
    print(f"model_b: {model_b}")
    print(f"input_shape: {shape_a}")
    if image_paths:
        print(f"input_mode: images from {Path(args.image_dir)}")
        print(f"samples: {len(image_paths)}")
    else:
        print("input_mode: random")
        print(f"samples: {args.samples}")
    print()

    max_abs_values: list[float] = []
    mean_abs_values: list[float] = []
    rmse_values: list[float] = []
    cosine_values: list[float] = []
    argmax_match_count = 0
    top5_exact_match_count = 0
    top5_overlap_values: list[int] = []

    sample_count = len(image_paths) if image_paths else args.samples
    for sample_index in range(sample_count):
        if image_paths:
            input_array = build_image_input(shape_a, image_paths[sample_index])
        else:
            input_array = build_random_input(shape_a, args.seed + sample_index)
        output_a = run_model(session_a, input_array)
        output_b = run_model(session_b, input_array)
        summary = summarize_diff(output_a, output_b)

        max_abs_values.append(float(summary["max_abs_diff"]))
        mean_abs_values.append(float(summary["mean_abs_diff"]))
        rmse_values.append(float(summary["rmse"]))
        cosine_values.append(float(summary["cosine_similarity"]))
        argmax_match_count += int(bool(summary["argmax_match"]))
        top5_exact_match_count += int(bool(summary["top5_exact_match"]))
        top5_overlap_values.append(int(summary["top5_overlap"]))

        print(
            f"[sample {sample_index:02d}] "
            f"max_abs_diff={summary['max_abs_diff']:.6f} "
            f"mean_abs_diff={summary['mean_abs_diff']:.6f} "
            f"rmse={summary['rmse']:.6f} "
            f"cosine={summary['cosine_similarity']:.6f} "
            f"argmax=({summary['argmax_a']},{summary['argmax_b']}) "
            f"match={summary['argmax_match']} "
            f"top5_exact={summary['top5_exact_match']} "
            f"top5_overlap={summary['top5_overlap']}/5"
        )

    print()
    print("summary:")
    print(
        f"  max_abs_diff   min/mean/max = "
        f"{min(max_abs_values):.6f} / {np.mean(max_abs_values):.6f} / {max(max_abs_values):.6f}"
    )
    print(
        f"  mean_abs_diff  min/mean/max = "
        f"{min(mean_abs_values):.6f} / {np.mean(mean_abs_values):.6f} / {max(mean_abs_values):.6f}"
    )
    print(
        f"  rmse           min/mean/max = "
        f"{min(rmse_values):.6f} / {np.mean(rmse_values):.6f} / {max(rmse_values):.6f}"
    )
    print(
        f"  cosine         min/mean/max = "
        f"{min(cosine_values):.6f} / {np.mean(cosine_values):.6f} / {max(cosine_values):.6f}"
    )
    print(f"  argmax_match   {argmax_match_count} / {sample_count}")
    print(f"  top5_exact     {top5_exact_match_count} / {sample_count}")
    print(
        f"  top5_overlap   min/mean/max = "
        f"{min(top5_overlap_values)} / {np.mean(top5_overlap_values):.3f} / {max(top5_overlap_values)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
