# 03 ONNX Optimization

## Concept Primer

This document is a CPU-oriented ONNX graph optimization after the initial CNN smoke test and power test. The current focus is the `ResNet50` INT8 model path used in earlier experiments.

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

### 1. Prepare the INT8 model

If the INT8 model has not already been generated, create it from the FP32 source model:

```powershell
wget -O models/resnet50.onnx https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub/resolve/main/resnet50_Opset17_torch_hub.onnx

python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50.onnx --quantized_model_path models/resnet50_A8W8.onnx --config A8W8

$runId = "onnx_op_optimization"
python tools/capture_env.py --run-id $runId

python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50.onnx --device cpu --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 -profile-out unoptimized_cpu_int8.json

```

### 2. Inspect the current graph structure

Open the profile files correspond to the INT8 `resnet50_A8W8.onnx` model on each device, starts with `onnxruntime_profile__*.json`

using https://www.ui.perfetto.dev/. 


### 3 Depply analyze the quantization workflow
Analyze AMD quark's quantization workflow 

### 4. Apply graph optimizations

### 5. Validate the optimizations
- Re-run the optimized graph and capture the new profile.
- Compare the new profile with the original to confirm that the expected optimizations are present and that the CPU execution time has improved.    


