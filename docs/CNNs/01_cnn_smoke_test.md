# 10 CNN Smoke Test

## Concept Primer

This docs describes CNN smoke testing before we scale up our evaluation. The goal is to validate our end-to-end capture and parsing workflows on a known workload with expected performance characteristics. We will run a small matrix of ResNet50 inference tests across FP32/FP16/INT8/BF16/FP16 precisions and NPU/iGPU/CPU devices, capturing latency and power metrics where possible.

1. Quantize ResNet50 FP32 model to other precisions using Quark.
2. Run inference with ONNX Runtime, capturing latency and power.


## Preconditions

0. Check the 00_setup.md and 00_tools.md docs for any required setup steps and tools.
1. Windows environment ready (`conda activate ryzenai170`) and Ryzen AI SDK installed.
2. ORT providers available:
   - `VitisAIExecutionProvider`
   - `DmlExecutionProvider`
   - `CPUExecutionProvider`
3. Source FP32 model is available for INT8/BF16 conversion.
4. HWINFO is installed and CSV logging is configured for power capture.
5. custom_ops.dll is compiled and accessible for ONNX Runtime.


## INT8/FP16/BF16 Model Preparation (Quark)

Used FP32 model is from https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub.

```powershell
wget -O models/resnet50.onnx https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub/resolve/main/resnet50_Opset17_torch_hub.onnx

python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50.onnx --quantized_model_path models/resnet50_A8W8.onnx --config A8W8

python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50.onnx --quantized_model_path models/resnet50_BF16.onnx --config BF16

python -m quark.onnx.tools.convert_fp32_to_bf16 --input models/resnet50.onnx --output models/resnet50_BF16_cast.onnx --format with_cast

python -m quark.onnx.tools.convert_fp32_to_fp16 --input models/resnet50.onnx --output models/resnet50_FP16.onnx --keep_io_types
```

## How To Run 

1. Capture environment details for this run:
```powershell
python tools/capture_env.py --run-id $runId
```

2. Capture benchmark execution with performance and power profiling:

```powershell
python tools/run_capture.py --run-id $runId -- python src/cnn_smoke_test.py --model-path $modelPath --precision $precision --device $device --input-shape 1x3x224x224 --batch 1 --repeat 30 --warmup 3 --profile-out "results/raw/$runId/ort_profile.json" --xrt-smi-out "results/raw/$runId/xrt_smi.txt" --hwinfo-csv "C:\path\to\hwinfo.csv" --require-power
```

## In detail explanation and expected results are in the `src/cnn_smoke_test.py` script, but in summary:
WIP