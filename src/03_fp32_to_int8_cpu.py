import argparse

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


# Dummy calibration data reader used to drive static quantization.
class DummyDataReader(CalibrationDataReader):
    def __init__(self, model_path: str) -> None:
        self.model = onnx.load(model_path)
        self.input_name = self.model.graph.input[0].name
        self.input_shape = [1, 3, 224, 224]
        self.data_count = 5
        self.current_idx = 0

    def get_next(self):
        if self.current_idx < self.data_count:
            self.current_idx += 1
            return {
                self.input_name: np.random.uniform(
                    -1.0, 1.0, self.input_shape
                ).astype(np.float32)
            }
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize an FP32 ONNX model into a CPU-targeted INT8 model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input FP32 ONNX model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output INT8 ONNX model.",
    )
    return parser


def quantize_model(input_path: str, output_path: str) -> None:
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=DummyDataReader(input_path),
        # Keep the QDQ format required by the CPU quantization flow.
        # It can also directly produce QLinear by QuantFormat.QOperator, but QDQ is more explicit and easier to debug.
        quant_format=QuantFormat.QDQ, 
        # Use UINT8 activations and INT8 weights for CPU VNNI execution.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = build_parser().parse_args()
    quantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
