#!/usr/bin/env python3
"""Rewrite the ResNet50 A8W8 QDQ graph with UINT8 activations and folded ReLUs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


@dataclass(frozen=True)
class RewriteStats:
    activation_nodes_rewritten: int
    activation_zero_points_created: int
    relu_nodes_folded: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a QDQ INT8 ONNX model by converting activation zero-points "
            "to UINT8 and folding Relu into the following QuantizeLinear."
        )
    )
    parser.add_argument(
        "--input",
        default="models/resnet50_A8W8.onnx",
        help="Path to the source ONNX model.",
    )
    parser.add_argument(
        "--output",
        default="models/resnet50_A8W8_CPU_from_int8.onnx",
        help="Path to save the rewritten ONNX model.",
    )
    return parser.parse_args()


def build_initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    return consumers


def add_initializer(
    graph: onnx.GraphProto,
    initializer_map: dict[str, onnx.TensorProto],
    name: str,
    array: np.ndarray,
) -> None:
    tensor = numpy_helper.from_array(array, name=name)
    graph.initializer.append(tensor)
    initializer_map[name] = tensor


def prune_unused_initializers(
    graph: onnx.GraphProto, initializer_map: dict[str, onnx.TensorProto]
) -> None:
    used_names = {input_name for node in graph.node for input_name in node.input}
    for initializer in list(graph.initializer):
        if initializer.name in used_names:
            continue
        graph.initializer.remove(initializer)
        initializer_map.pop(initializer.name, None)


def make_unique_name(
    initializer_map: dict[str, onnx.TensorProto], base_name: str
) -> str:
    if base_name not in initializer_map:
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in initializer_map:
            return candidate
        suffix += 1


def rewrite_activation_zero_points(
    graph: onnx.GraphProto,
    initializer_map: dict[str, onnx.TensorProto],
) -> tuple[int, int]:
    replacement_names: dict[str, str] = {}
    rewritten_nodes = 0

    for node in graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 3:
            continue
        if node.input[0] in initializer_map:
            continue

        zero_point_name = node.input[2]
        zero_point_tensor = initializer_map.get(zero_point_name)
        if zero_point_tensor is None:
            continue
        if zero_point_tensor.data_type != onnx.TensorProto.INT8:
            continue

        replacement_name = replacement_names.get(zero_point_name)
        if replacement_name is None:
            zero_point_array = numpy_helper.to_array(zero_point_tensor)
            replacement_array = (
                zero_point_array.astype(np.int16) + np.int16(128)
            ).astype(np.uint8)
            replacement_name = make_unique_name(
                initializer_map, f"{zero_point_name}_uint8"
            )
            add_initializer(graph, initializer_map, replacement_name, replacement_array)
            replacement_names[zero_point_name] = replacement_name

        node.input[2] = replacement_name
        rewritten_nodes += 1

    return rewritten_nodes, len(replacement_names)


def fold_relu_into_quantize(
    graph: onnx.GraphProto,
    initializer_map: dict[str, onnx.TensorProto],
) -> int:
    consumers = build_consumer_map(graph)
    nodes_to_remove: list[onnx.NodeProto] = []
    folded_relu_count = 0

    for relu_node in list(graph.node):
        if relu_node.op_type != "Relu":
            continue
        if len(relu_node.input) != 1 or len(relu_node.output) != 1:
            continue

        relu_output_name = relu_node.output[0]
        relu_consumers = consumers.get(relu_output_name, [])
        if len(relu_consumers) != 1:
            continue

        quantize_node = relu_consumers[0]
        if quantize_node.op_type != "QuantizeLinear":
            continue
        if len(quantize_node.input) < 3 or len(quantize_node.output) != 1:
            continue

        quantized_output_name = quantize_node.output[0]
        quantized_consumers = consumers.get(quantized_output_name, [])
        if not quantized_consumers:
            continue
        if any(node.op_type != "DequantizeLinear" for node in quantized_consumers):
            continue

        zero_point_name = quantize_node.input[2]
        zero_point_tensor = initializer_map.get(zero_point_name)
        if zero_point_tensor is None:
            continue
        if zero_point_tensor.data_type != onnx.TensorProto.UINT8:
            continue

        zero_point_array = numpy_helper.to_array(zero_point_tensor)
        if not np.all(zero_point_array == 128):
            continue

        folded_zero_point_name = make_unique_name(
            initializer_map, f"{zero_point_name}_relu_folded"
        )
        folded_zero_point_array = np.zeros_like(zero_point_array, dtype=np.uint8)
        add_initializer(
            graph,
            initializer_map,
            folded_zero_point_name,
            folded_zero_point_array,
        )

        # Keep the original scale so the folded path stays as close as possible
        # to the source model for non-saturated positive activations.
        quantize_node.input[0] = relu_node.input[0]
        quantize_node.input[2] = folded_zero_point_name

        for dequantize_node in quantized_consumers:
            if len(dequantize_node.input) < 3:
                continue
            dequantize_node.input[2] = folded_zero_point_name

        nodes_to_remove.append(relu_node)
        folded_relu_count += 1

    for node in nodes_to_remove:
        graph.node.remove(node)

    return folded_relu_count


def rewrite_model(input_path: Path, output_path: Path) -> RewriteStats:
    if not input_path.exists():
        raise SystemExit(f"Model file not found: {input_path}")

    model = onnx.load(str(input_path))
    graph = model.graph
    initializer_map = build_initializer_map(graph)

    activation_nodes_rewritten, activation_zero_points_created = (
        rewrite_activation_zero_points(graph, initializer_map)
    )
    relu_nodes_folded = fold_relu_into_quantize(graph, initializer_map)

    prune_unused_initializers(graph, initializer_map)
    graph.ClearField("value_info")
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return RewriteStats(
        activation_nodes_rewritten=activation_nodes_rewritten,
        activation_zero_points_created=activation_zero_points_created,
        relu_nodes_folded=relu_nodes_folded,
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    stats = rewrite_model(input_path=input_path, output_path=output_path)

    print(f"input_model: {input_path}")
    print(f"output_model: {output_path}")
    print(f"activation_nodes_rewritten: {stats.activation_nodes_rewritten}")
    print(f"activation_zero_points_created: {stats.activation_zero_points_created}")
    print(f"relu_nodes_folded: {stats.relu_nodes_folded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
