# 01 CNN Smoke Test

## Concept Primer
This document describes the CNN smoke test we run before scaling up the evaluation. The goal is to validate our end-to-end capture and parsing workflows on a known workload with expected performance characteristics. We run a small matrix of ResNet50 inference tests across FP32, FP16, and INT8 precisions on the NPU, iGPU, and CPU, capturing latency and profiling data with ONNX Runtime and Ryzen AI SDK tools.

1. Convert the ResNet50 FP32 model to other precisions using Quark.
2. Run inference with ONNX Runtime while capturing latency and profiling data.

## Preconditions

0. Check the `00_setup.md` and `01_tools.md` documents for required setup steps and tools.
1. The Windows environment is ready (`conda activate ryzenai171`), and the Ryzen AI SDK is installed.
2. The following ORT providers are available:
   - `VitisAIExecutionProvider`
   - `DmlExecutionProvider`
   - `CPUExecutionProvider`
3. The source FP32 model is available for FP16 and INT8 conversion.

## FP16/INT8 Model Preparation (Quark)

The FP32 source model used here is from https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub.

```powershell
wget -O models/resnet50.onnx https://huggingface.co/onnxmodelzoo/resnet50_Opset17_torch_hub/resolve/main/resnet50_Opset17_torch_hub.onnx

python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50.onnx --quantized_model_path models/resnet50_A8W8.onnx --config A8W8

python -m quark.onnx.tools.convert_fp32_to_fp16 --input models/resnet50.onnx --output models/resnet50_FP16.onnx --keep_io_types
```

## How to Run

1. Capture environment details for this run:

```powershell
$runId = "cnn_smoke_test"
python tools/capture_env.py --run-id $runId
```

2. Capture benchmark execution with performance profiling.

Manual run:

```powershell
python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50.onnx --device npu --disable-fallback --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 --vaip-cache-dir results/raw/$runId --vaip-cache-key resnet50_FP32
```

Recommended automated run:

```powershell
& .\tools\01_cnn_smoke_test.ps1
```

3. Extract the structured results from `stdout.log` into CSV:

```powershell
python src/CNNs/01_extract_cnn_smoke_results.py --stdout-log results/raw/$runId/stdout.log --metadata-log results/raw/$runId/metadata.jsonl --csv-out results/raw/$runId/cnn_smoke_results.csv
```

4. Aggregate duplicate rows from `cnn_smoke_results.csv` into a summary CSV:

```powershell
python src/CNNs/01_sumary_extract_from_csv.py --csv-in results/raw/$runId/cnn_smoke_results.csv --csv-out results/raw/$runId/cnn_smoke_results_summary.csv
```

## Observed Results

- `results/raw/$runId/cnn_smoke_results.csv`
- `results/raw/$runId/cnn_smoke_results_summary.csv`
- `stdout.log`, `stderr.log`, `metadata.jsonl`, and eight `onnxruntime_profile__*.json` files under `results/raw/$runId/` for this smoke test
- One failed ONNX Runtime profiling trace may also be written to the current working directory

### 1. Profiling On vs. Off

| Model | Device | Profiling Off Mean (ms) | Profiling On Mean (ms) | Delta (ms) | Delta (%) |
|---|---|---:|---:|---:|---:|
| `resnet50.onnx` | CPU | 7.5120 | 10.0344 | 2.5224 | 33.58 |
| `resnet50.onnx` | NPU | 5.6539 | 5.6356 | -0.0183 | -0.32 |
| `resnet50.onnx` | iGPU | 2.8631 | 2.8122 | -0.0509 | -1.78 |

For this set of runs, enabling profiling increased latency for most successful combinations. The largest overhead appeared on CPU for `resnet50.onnx`. Strictly speaking, this is not a one-to-one comparison, because profiling-enabled runs were executed once, whereas profiling-disabled runs were executed ten times each. Even so, the data still suggests that profiling introduces overhead, with a particularly strong effect on CPU. On NPU and iGPU, the impact was relatively small. This may indicate either that profiling consumes more CPU-side resources or that CPU execution is simply more sensitive to profiling overhead. Another plausible explanation is that CPU runs expose many more nodes to the profiler, which increases collection cost. This experiment therefore reinforces a simple rule: disable profiling when measuring raw latency. If profiling is necessary, interpret the results with its latency impact in mind.

### 2. CPU vs. NPU vs. iGPU, Profiling Off

| Model | CPU Mean (ms) | NPU Mean (ms) | iGPU Mean (ms) | Fastest Device |
|---|---:|---:|---:|---|
| `resnet50.onnx` | 7.5120 | 5.6539 | 2.8631 | iGPU |
| `resnet50_FP16.onnx` | 7.1600 | null | 1.3794 | iGPU |
| `resnet50_A8W8.onnx` | 26.4818 | 3.1781 | 2.3619 | iGPU |

iGPU was the fastest successful device for all three model variants. NPU remained faster than CPU for the FP32 and INT8 models. The FP16 failure is discussed in more detail in Section 3. The fact that the INT8 model was slower than FP32 on CPU is not surprising and may indicate that the CPU path is not optimized for INT8 execution. On NPU, the INT8 model was faster than FP32, likely because INT8 reduces memory-bandwidth requirements and because the NPU is optimized for INT8 operations. We also confirmed that the FP32 model runs as BF16 on NPU; this is discussed in more detail in Section 3. Given the NPU architecture and optimization characteristics, INT8 can be significantly faster than FP32 for models with many MAC operations.

On iGPU, the FP16 model delivered the best performance, which may indicate that the iGPU is well optimized for FP16 computation. In particular, FP16 reduces data size and can increase effective throughput relative to FP32, which helps lower latency. These results suggest that the effect of model precision on performance varies by device architecture and optimization level. They reaffirm the importance of choosing the right data type for each architecture.

### 3. FP16 NPU Run Failure Analysis

The FP16 model did not run on the NPU because CPU fallback was explicitly disabled. By default, ONNX Runtime falls back to CPU when the model encounters an operation unsupported on the NPU. In our configuration, however, we forced the run to fail whenever fallback occurred by using the following code.

```python
session_options.add_session_config_entry(
   "session.disable_cpu_ep_fallback",
   "1"
)

session = ort.InferenceSession(
...
   sess_options=session_options,
   providers=[config.provider],
)

session.disable_fallback()
```

To identify which nodes triggered fallback, run the following commands. We re-enabled fallback and logged the device assignment for each node.

```powershell
$env:XLNX_ONNX_EP_REPORT_FILE = "vitisai_ep_report.json"
python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50_FP16.onnx --device npu --input-shape 3x224x224 --batch 1 --warmup 1 --repeat 1 --vaip-cache-dir results/raw/$runId --vaip-cache-key resnet50_FP16_Fallback --clear-vaip-cache
```

Then open `results/raw/$runId/resnet50_FP16_Fallback/vitisai_ep_report.json` and check which nodes fell back to CPU. In this case, every node was assigned to CPU, which is the expected result. The official Ryzen AI repository states:

```
The Ryzen AI compiler supports input models in the following formats:

CNN Models

- INT8 (quantized)
- FP32 (automatically converted to BF16 during compilation)

...
Ryzen AI Software natively supports both CNN and Transformer models in floating-point (FP32) format. When FP32 models are provided as input, the VitisAI EP automatically converts them to bfloat16 (BF16) precision and processes them through the optimized BF16 compilation pipeline.
```

This confirms that FP16 is not supported and that the FP32 model runs as BF16 on the NPU. In practice, compiling the FP32 model on NPU takes noticeably longer because the compilation step includes FP32-to-BF16 conversion. You can also confirm this from an intermediate file generated during model cache creation: `results\raw\cnn_smoke_test\FP32_preliminary-vaiml-pass-summary.txt` shows that the model is recognized as FP32 and that the device adopts BF16 as the working data type. Therefore, for CNN workloads on the NPU today, use FP32 input when accuracy is the priority, since it is converted to BF16 internally, or use INT8 when performance is the priority.

### 4. ONNX Runtime Profiling Data

The ONNX Runtime profiling data generated with the profiling option can be viewed in https://www.ui.perfetto.dev/, as described at https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html.

Open the profile files correspond to the INT8 `resnet50_A8W8.onnx` model on each device, starts with `onnxruntime_profile__*.json`

When you open these files, you will see one `session_initialization` event for session creation, 60 `model_run` events (10 warm-up runs and 50 measured runs), and detailed node-level information under each `model_run`.

#### A8W8 CPU Profile Analysis

First, revisit the result from Section 1. On CPU, the model was not well optimized and executed as many small, fragmented nodes such as `QuantizeLinear`, `DequantizeLinear`, `FusedConv`, and `Transpose`. The ONNX Runtime profiler has to trace each of those nodes, so the overhead accumulates. By contrast, on NPU (`vitis_ai_ep_1`) and iGPU (`DmlFusedNode_0_0`), the entire model graph was merged into a single large fused node. From the profiler's perspective, it only needs to time two events: dispatching work to the hardware and receiving completion. The difference in the number of events that must be traced is substantial.

The main cause of the CPU bottleneck appears to be that this model is not compiled into a truly INT8-native graph. Instead, it is represented with Q-DQ (`QuantizeLinear`-`DequantizeLinear`) nodes. We also observed that the large number of `Transpose` nodes slows CPU execution further. To try to remove these `Transpose` nodes, we used Quark to convert the FP32 model input format from NCHW to NHWC.

```powershell
python -m quark.onnx.tools.convert_nchw_to_nhwc --input .\models\resnet50.onnx --output .\models\resnet50_NHWC.onnx
```

However, after inspecting the model in Netron, it turned out that the tool simply added a `Transpose` node that accepts the input in NHWC and then converts it back to NCHW. A proper NCHW-to-NHWC conversion would also need to update the input and output layouts of the internal operators. That did not happen with this tool, so a different library will likely be needed. That investigation is outside the scope of this document.

Finally, inspect the inputs, outputs, and surrounding ops for convolution-related nodes such as `FusedConv` and `QLinearConv`. Their inputs differ in layout (NCHW vs. NHWC) and type (INT vs. FP), which confirms that `QDQ` and `Transpose` nodes were inserted throughout the graph. It appears necessary to consolidate these operations as much as possible.

For reference, Perfetto SQL:

```sql
SELECT
  args.string_value AS op_name,
  COUNT(slice.id) AS call_count,
  SUM(slice.dur) / 1000000.0 AS total_time_ms
FROM slice
JOIN args ON slice.arg_set_id = args.arg_set_id
WHERE args.key = 'args.op_name'
GROUP BY args.string_value
ORDER BY total_time_ms DESC
```

#### A8W8 NPU Profile Analysis

Now consider the NPU profile. In the CPU profile, hundreds of nodes such as `FusedConv`, `Relu`, and `Add` were executed in fragmented form. On NPU, however, the entire model was fused into a single custom node, `vitis_ai_ep_1`. This indicates that most of the ResNet50 computation was successfully offloaded to the NPU (DPU). The `QuantizeLinear` / `DequantizeLinear` operations that consumed a large amount of execution time on CPU almost disappeared on NPU, accounting for only about 1% each. The remaining QDQ calls are only the minimal I/O conversions needed to convert the initial input to INT8 and the final output back to FP32. One important point is model loading and initialization time. The `session_initialization` phase includes converting the ONNX graph into a form that the NPU can understand, and this takes a long time. For repeated runs with fixed models, inputs, and parameters, it therefore makes sense to rely on model caching.

#### A8W8 iGPU Profile Analysis

For iGPU (DirectML), there is no visible QDQ overhead and model initialization time is also short, which is encouraging. The absence of QDQ suggests that the necessary FP32 handling may be absorbed into the fused iGPU execution path. However, the detailed data shows something interesting: the pure execution time of `DmlFusedNode_0_0` itself is very short, but the total `model_run` time is much longer. This implies that most of the inference time is spent copying and synchronizing input and output tensors between CPU and iGPU. That is a clear optimization opportunity. Concretely, the input is currently a CPU NumPy array, so for non-CPU execution providers the measured time likely includes host-to-device and device-to-host copy costs. Using `io_binding` to place the input tensor directly in iGPU memory could improve end-to-end performance. (https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs)

#### FP32 NPU Profile Analysis

Now consider the NPU profiling data for the FP32 model. Open a profile file correspond to the FP32 `resnet50.onnx` model on NPU

This file shows that model initialization is extremely long. This is because, once the FP32 model is detected, session initialization has to convert FP32 to BF16. This again highlights the importance of using the right data type for the target accelerator.

### 5. Warm-up and Caching

As shown above, model initialization on NPU takes a long time, and other profiling data also shows cases where the first run is slower. That is why model caching and warm-up matter. Model caching stores an optimized form of the model during the first execution. Subsequent runs can then load that optimized artifact directly, which can significantly reduce initialization time. Likewise, warm-up runs allow the model and hardware to reach a stable operating state before measurement begins, reducing first-run artifacts and producing more consistent performance numbers. In Ryzen AI, model caching is enabled by default, but cached artifacts are not automatically reused across all parameter changes, so this behavior must be handled carefully.

### Next Steps

We observed that the iGPU outperformed the NPU in this smoke test. The next question is power consumption. We should analyze the power profiling data captured during this smoke test to determine whether there are meaningful differences in power usage across devices and model precisions. That will help us understand the energy-efficiency tradeoffs of running these models on different hardware.
