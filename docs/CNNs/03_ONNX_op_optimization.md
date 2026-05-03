# 03 ONNX Optimization

## Concept Primer

This document covers CPU-oriented ONNX graph optimization following the initial CNN smoke test and power test. The current focus is the `ResNet50` INT8 model path used in earlier experiments.

The motivation is straightforward. In prior CPU runs, the quantized model did not execute as a clean INT8-native graph. Instead, the graph included a mixture of `QuantizeLinear`, `DequantizeLinear`, `FusedConv`, and related operators. A large number of `Transpose` nodes were also visible. These patterns are plausible sources of CPU-side overhead.

## Scope

- Revisit the INT8 `ResNet50` ONNX graph used in the earlier CNN experiments.
- Inspect whether the CPU path is dominated by `QDQ`-style graph fragments rather than a more consolidated INT8 execution pattern.
- Check whether layout conversion or graph rewriting can reduce unnecessary `Transpose` operations.

## Preconditions

0. Review `docs/00_setup.md`, `docs/01_tools.md`, and `docs/CNNs/01_cnn_smoke_test.md`.
1. The Windows environment is ready and the required Quark and ONNX Runtime tooling is available.
2. `models/resnet50.onnx` is available as the FP32 source model.
3. Any optimization claim must be validated from captured logs and saved artifacts rather than manual inspection alone.

## How to Run

### 1. Prepare the INT8 model and run the unoptimized baseline

If the INT8 model has not already been generated, create it from the FP32 source model:

```powershell
wget -O models/resnet50.onnx https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub/resolve/main/resnet50_Opset17_torch_hub.onnx

python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50.onnx --quantized_model_path models/resnet50_A8W8.onnx --config A8W8

$runId = "onnx_op_optimization"
python tools/capture_env.py --run-id $runId

python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50_A8W8.onnx --device cpu --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 --profile-out results/raw/$runID/unoptimized_cpu_int8.json
```

### 2. Inspect the current graph structure

Open the profile file corresponding to the INT8 `resnet50_A8W8.onnx` model on the CPU, specifically `unoptimized_cpu_int8.json`, in https://www.ui.perfetto.dev/.

As observed in the earlier smoke test, the graph is dominated by `QDQ`-style patterns and a large number of `Transpose` nodes. This suggests that the quantization workflow is not fully optimized for CPU execution and that there may be opportunities to consolidate operations and reduce overhead.


### 3. Deeply analyze the quantization workflow

Analyze AMD Quark's quantization workflow. It is open source on GitHub at https://github.com/amd/Quark/tree/release/0.11.

The CLI entry points lead to the following workflow:

`random_quantize.py` (`Quark-release-0.11\quark\onnx\tools\random_quantize.py`, line 56) loads the preset configuration named `A8W8`, forces `extra_options["UseRandomData"] = True`, leaves `calib_datareader = None`, wraps the configuration in the older `Config(...)` API, and calls `ModelQuantizer.quantize_model(...)`.
This eventually reaches `Quark-release-0.11\quark\onnx\quantization\quantize.py`, where `quantize_static()` (line 91) quantizes the model.

The `A8W8` configuration is defined as follows:
`custom_config.py` (line 290) defines `A8W8` as:
calibrate_method = MinMax
quant_format = ExtendedQuantFormat.QDQ
activation_type = QInt8
weight_type = QInt8
extra_options = {"ActivationSymmetric": True, "AlignSlice": False, "FoldRelu": True, "AlignConcat": True ...}
The string name is resolved by `get_default_config()` (line 943).

There is nothing particularly unusual here. The workflow inserts QDQ pairs around eligible nodes and applies MinMax quantization with both activations and weights symmetrically quantized to QInt8 under the A8W8 preset.

### 4. Apply graph optimizations

#### 4.1 Quantize the FP32 model into an INT8 model
- Use `03_fp32_to_int8_cpu.py` to convert the FP32 model to an INT8 model optimized for CPU execution. This script applies specific quantization parameters and graph transformations to better align with CPU execution patterns. The key transformations include UINT8 activations, INT8 weights, and QDQ quantization. The model can be converted directly to `QLinear` form with `QuantFormat.QOperator`, but QDQ is used here to check the CPU path's QDQ-to-`QLinear` conversion and to make debugging easier.
- After conversion, profile the optimized model using the same smoke test script to generate a new profile file for comparison.
```powershell
 python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50_A8W8_CPU_from_fp32.onnx --device cpu --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 --profile-out results/raw/$runID/resnet50_A8W8_CPU_from_fp32.json
```
- Open the generated profile file, `resnet50_A8W8_CPU_from_fp32.json`, and inspect the graph structure. Check for the presence of `QLinearConv` nodes, which indicate successful quantized convolution conversion, and verify that the number of `Transpose` nodes has been reduced compared with the unoptimized version.

In detail:
1. Previous unoptimized `QDQ + FusedConv` pattern is converted into `QLinearConv`.
2. `QLinearConv` nodes have NHWC Layout and UINT8/INT32 input and UINT8 output
3. For `Transpose`, the remaining nodes can be observed only at the beginning of the graph and around the global average pooling stage near the end. This suggests that converting the model to an NHWC layout may also remove the initial `Transpose`. For example,
```powershell
python -m quark.onnx.tools.convert_nchw_to_nhwc --input .\models\resnet50.onnx --output .\models\resnet50_NHWC.onnx
```
Using this command adds a `Transpose` at the model input, allowing the runtime to eliminate the explicit transpose during execution and consume the appropriately transformed input.

- Using Netron or similar ONNX graph visualization tools, inspect the optimized model's graph structure. Look for the following changes compared to the original unoptimized INT8 model:

1. Activation nodes are UINT8.
2. The `DQ -> Conv -> Relu -> Q` pattern changes to `DQ -> Conv -> Quantize`. Because the activation type is UINT8, negative values cannot be represented, so `Relu` is effectively folded into the quantization step. This folded form also appears to make it easier for the CPU path to convert QDQ segments into `QLinearConv`. When the `Conv -> Relu` pattern remains explicit, it seems to be fused only as `FusedConv` rather than lowered into a quantized convolution chain.

- From a latency perspective, the sample count of 50 runs is not especially large, so this should be treated cautiously. Even so, the optimized model drops to roughly 3 to 4 seconds, a substantial reduction relative to the unoptimized model, which took more than 20 seconds.
