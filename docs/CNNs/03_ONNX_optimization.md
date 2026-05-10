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

```text
calibrate_method = MinMax
quant_format = ExtendedQuantFormat.QDQ
activation_type = QInt8
weight_type = QInt8
extra_options = {"ActivationSymmetric": True, "AlignSlice": False, "FoldRelu": True, "AlignConcat": True ...}
```

The string name is resolved by `get_default_config()` (line 943).

There is nothing particularly unusual here. The workflow inserts QDQ pairs around eligible nodes and applies MinMax quantization with both activations and weights symmetrically quantized to QInt8 under the A8W8 preset.

### 4. Apply graph optimizations

#### 4.1 Quantize the FP32 model into an INT8 model

Use `03_fp32_to_int8_cpu.py` to convert the FP32 model to an INT8 model optimized for CPU execution. This script applies specific quantization parameters and graph transformations to better align with CPU execution patterns. The key transformations include UINT8 activations, INT8 weights, and QDQ quantization. The model can be converted directly to `QLinear` form with `QuantFormat.QOperator`, but QDQ is used here to check the CPU path's QDQ-to-`QLinear` conversion and to make debugging easier.

After conversion, profile the optimized model using the same smoke test script to generate a new profile file for comparison.

```powershell
python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50_A8W8_CPU_from_fp32.onnx --device cpu --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 --profile-out results/raw/$runID/resnet50_A8W8_CPU_from_fp32.json
```

##### 4.1.1 Profile data analysis
Open the generated profile file, `resnet50_A8W8_CPU_from_fp32.json`, and inspect the graph structure. Check for the presence of `QLinearConv` nodes, which indicate successful quantized convolution conversion, and verify that the number of `Transpose` nodes has been reduced compared with the unoptimized version.

In detail:

1. The previous fragmented Conv/Relu/QDQ execution path is reduced to `QLinearConv` for Conv groups whose activation clip is redundant and can be selected by the QDQ optimizer.
2. `QLinearConv` nodes use NHWC layout, with `UINT8` activations, `INT8` weights, optional `INT32` bias, and `UINT8` output.
3. The remaining `Transpose` nodes appear only at the beginning of the graph and around the global average pooling stage near the end. This suggests that converting the model to an NHWC layout may also remove the initial `Transpose`. For example:

```powershell
python -m quark.onnx.tools.convert_nchw_to_nhwc --input .\models\resnet50.onnx --output .\models\resnet50_NHWC.onnx
```

Using this command adds a `Transpose` at the model input, allowing the runtime to eliminate the explicit transpose during execution and consume the appropriately transformed input.

##### 4.1.1 Netron (visualization) analysis
Using Netron or similar ONNX graph visualization tools, inspect the optimized model's graph structure. Look for the following changes compared to the original unoptimized INT8 model:

1. Activation nodes are `UINT8`.
2. The `DQ -> Conv -> Relu -> Q` pattern changes to `DQ -> Conv -> Quantize` when the quantization parameters make the `Relu` redundant. The key condition is not `UINT8` by itself, but that the quantized lower bound maps to real value `0`.


From a latency perspective, the sample count of 50 runs is not especially large, so this should be treated cautiously. Even so, the optimized model drops to roughly 3 to 4 seconds, a substantial reduction relative to the unoptimized model, which took more than 20 seconds.

##### Why use weight `INT8`, activation `UINT8`, and `INT32` bias?

1. CPU architecture and backend libraries

ONNX Runtime CPU quantization supports multiple 8-bit signedness choices, including U8U8, U8S8, and S8S8. The official default recommendation is still S8S8 with QDQ because it balances performance and accuracy. U8S8 should therefore be treated as a CPU-oriented rewrite candidate, not as a universally better format. Its usefulness depends on the CPU, ONNX Runtime build, quantization format, and measured accuracy.

2. Quantization

Weights are static data produced after training and are usually distributed roughly symmetrically around zero, which makes symmetric quantization (`INT8`) easy to apply. Activations, in contrast, change dynamically with the input data and often have asymmetric distributions. In addition, tensors that pass through `ReLU` or similar activation functions do not contain negative values. Quantizing those tensors to `INT8` (`-128` to `127`) wastes roughly half of the representable range on negative values that are never used.

3. INT32 bias

`GEMM` is fundamentally a repeated sequence of multiplication and addition. Multiplying two 8-bit integers produces up to a 16-bit intermediate value (`2^8 * 2^8 = 2^16`). In convolution or linear layers, thousands of these products must then be accumulated into a single result. A 16-bit accumulator would overflow quickly during that process, so hardware typically uses a 32-bit accumulator (`INT32`) to hold the partial sums safely. Because bias is added after accumulation, using `INT32` bias is the most natural system-level choice and avoids extra casting overhead.

#### 4.2 Graph rewriting for optimization

```powershell
python src/CNNs/03_rewriting_ONNX_from_A8W8.py --input .\models\resnet50_A8W8.onnx --output .\models\resnet50_A8W8_uint8withFold.onnx
```

Specifically, the following optimizations are applied.

1. From the `resnet_A8W8` model, change activation `INT8` to `UINT8`.

To identify activation layers, consider the following example:

```text
node = DequantizeLinear(x, scale, zero_point)
x = input[0]
scale = input[1]
zero_point = input[2]
```

Here, `initializer` refers to a fixed tensor stored in the model file, such as weight, bias, scale, or zero-point. If `node.input[0] == initializer`, the target tensor processed by the Q/DQ pair (`x`) is a constant tensor. That usually indicates a weight Q/DQ pair. Conversely, if `input[0]` is not an initializer, it can usually be inferred as an activation Q/DQ pair, although exceptions are still possible.

For nodes inferred to be activation Q/DQ pairs, find their `zero_point` values and convert them as follows:

```python
replacement_array = (
        zero_point_name = node.input[2]
        ... (load the zero_point array from the initializer)
    zero_point_array.astype(np.int16) + np.int16(128)
).astype(np.uint8)
```

This extracts the ONNX tensor into a NumPy array, adds `128` to the `INT8` values to shift them into the `UINT8` range, and uses a temporary `int16` cast to avoid overflow during the intermediate computation.

2. Fold `ReLU` into quantization.

Find the `Relu -> QuantizeLinear -> DequantizeLinear` pattern and remove `Relu`.

The removable `Relu` must satisfy all of the following conditions:

1. `Relu` is a normal single-input, single-output node (lines 139-143).
2. The output of `Relu` is consumed by exactly one node (lines 145-148).
3. That consumer is `QuantizeLinear` (lines 150-154).
4. The output of that `QuantizeLinear` is consumed only by `DequantizeLinear` nodes (lines 156-161).
5. The `zero-point` of `QuantizeLinear` is `UINT8`, and all values are `128` (lines 163-172).

The last condition is important. Before folding, this pattern uses the shifted `UINT8` representation of the former symmetric `INT8` activation, so real zero is represented by `zero_point = 128`. The fold is valid only because the rewrite then bypasses `Relu` and changes the following Q/DQ zero point to `0`, making the `UINT8` lower bound represent real zero.

In other words, the negative-value suppression previously performed by `Relu` is replaced by lower-bound clipping in `UINT8 QuantizeLinear(zp=0)`, where `qmin = 0` dequantizes to real value `0`. The saturation behavior on the large positive side can differ.

The equivalence for negative inputs is straightforward:

```text
Original path:
Relu -> Quantize(zp=128) -> Dequantize(zp=128)

x_relu = max(x, 0)
q = round(x_relu / scale) + 128
y = (q - 128) * scale

Folded path:
Quantize(zp=0) -> Dequantize(zp=0)

q = saturate(round(x / scale) + 0)
y = q * scale
```

For `x < 0`, the original path produces `x_relu = 0`, so no negative value reaches quantization. In the folded path, `zero_point = 0` makes the `UINT8` lower bound map to real zero, so the quantized value is clipped to `0`. In both cases, the final dequantized result is `0`.

For example, with `scale = 0.1` and `x = -0.3`:

```text
Original:
Relu(-0.3) = 0
q = round(0 / 0.1) + 128 = 128
y = (128 - 128) * 0.1 = 0

Folded:
q = saturate(round(-0.3 / 0.1) + 0)
  = saturate(-3)
  = 0
y = 0 * 0.1 = 0
```

So, for the negative-input case that `ReLU` handles, the folded form preserves the same effect.

Through this transformation, the `Conv -> Relu -> Quantize` pattern becomes `Conv -> Quantize`. The original post-Relu `UINT8` quantizer uses `zero_point = 128` to preserve the shifted symmetric range, but the folded form uses `zero_point = 0` so that the quantizer's lower bound implements the negative-value suppression effect of `ReLU`. This is the same condition ONNX Runtime checks internally for treating `Relu` as a redundant clip. In addition, this folded form lets the CPU optimizer select the Conv QDQ group and convert it into `QLinearConv`. In contrast, when `ReLU` remains explicit because the redundant-clip condition is not met, the Conv QDQ group is not selected and the remaining float-domain `Conv + Relu` can be fused into `FusedConv`.

#### 4.3 Validate the optimized graph's performance

```powershell
python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50_A8W8_uint8withFold.onnx --device cpu --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 50 --profile-out results/raw/$runID/resnet50_A8W8_CPU_uint8withFold.json
```

Use the profiler and Netron to verify that the intended changes actually occurred. For example, check whether the `Conv -> Relu -> Quantize` pattern changed to `Conv -> Quantize`, whether `QLinearConv` nodes appeared, and whether the number of `Transpose` nodes decreased. Also verify directly from the initializers that the `zero-point` values were changed correctly to `128` or `0`.

Compared with the fully optimized `resnet50_A8W8_CPU_from_fp32.onnx` model, the number of `QLinearConv` nodes is the same. In other words, the graph changes we intended were applied successfully. However, when latency was compared, there was still a difference of about 3 ms. A closer inspection in Pallette showed that overhead was still being introduced by transposes around `maxpool` and by operations such as `ReorderInput_kernel_time`. This session does not address that issue further, but it indicates that the graph layout is still not fully understood or optimized.


Also compare whether the model outputs before and after the transformation are similar.

```powershell
python .\src\CNNs\03_compare_two_cpu_models.py `
  --model-a .\models\resnet50_A8W8.onnx `
  --model-b .\models\resnet50_A8W8_uint8withFold.onnx `
  --samples 1000
  --seed 42
```

These two models were quantized with random data, so they should not be treated as especially stable models. In addition, transformations such as folding are not mathematically identical in every case, so differences between the two outputs can appear. Even so, it is still useful to check whether they remain broadly similar.

Summary:

```text
max_abs_diff   min/mean/max = 0.312202 / 0.745344 / 1.834189
mean_abs_diff  min/mean/max = 0.047260 / 0.084908 / 0.124920
rmse           min/mean/max = 0.068833 / 0.122779 / 0.179249
cosine         min/mean/max = 0.979707 / 0.990509 / 0.997217
argmax_match   952 / 1000
top5_exact     12 / 1000
top5_overlap   min/mean/max = 2 / 3.678 / 5
```

There is some localized numerical error (`max_abs_diff`), but the overall distribution (`cosine`) is nearly the same. Most importantly, the final prediction (`argmax_match`) matches more than 95% of the time, so the transformed model can be regarded as producing broadly similar predictions to the original model. The low `top5_exact` value may be due to changes in quantization characteristics during folding, which can alter the ranking of some classes. In addition, the average `top5_overlap` is `3.678`, showing that although some of the top predicted classes changed because of quantization and folding, there is still substantial overlap. Therefore, the overall conclusion is that the model's predictions did not change dramatically.

#### 4.4 Deep dive "why" Conv+ReLU is converted FusedConv or converted to QLinearConv

The key distinction is whether `Relu` can be treated as a redundant clipping node.

For the `DQ -> Conv -> Relu -> Q` pattern, ONNX Runtime first tries to select a QDQ Conv group. In `qdq_selectors.cc`, the selector detects a following `Relu` or `Clip` as a possible redundant clip node. However, it accepts that node only if `IsClipMadeRedundantByQ(...)` returns true. If the `Relu` is not redundant, the selector returns `std::nullopt`, so the Conv QDQ group is not selected.

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.cc
- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/qdq_util.cc

`IsClipMadeRedundantByQ(...)` defines the `Relu` condition directly. For `Relu`, the function returns true only when the output `QuantizeLinear` zero point is equal to the minimum value of the quantized data type:

```cpp
if (clip_op_type == "Relu") {
  return zp == data_type_min;
}
```

For example, with `int8`, `data_type_min` is `-128`. Therefore, `int8` activation quantization with `zero_point = -128` makes the lower quantization bound represent real value `0`. Negative values are already clipped by `QuantizeLinear`, so `Relu` is redundant. With symmetric `int8 zero_point = 0`, negative values remain representable, so `Relu` is not redundant.

If the QDQ Conv group is selected, `qdq_actions.cc` replaces it with `QLinearConv`. The Conv replacement action moves the input DQ parameters, weight DQ parameters, output Q scale, output Q zero point, and optional bias into the new QLinear operator. There is no separate path here that creates `QLinearConv -> Relu` for this pattern.

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_actions.cc

When `Relu` is not redundant, the QDQ Conv selector fails. The graph still contains a float-domain `Conv + Relu` region between QDQ nodes. Later, `ConvActivationFusion` can select that `Conv` if its only consumer is a supported activation such as `Relu`. The fusion action changes ONNX `Conv` into `com.microsoft.FusedConv` and stores the activation type as an attribute.

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/conv_activation_fusion.cc

The optimizer registration order also supports this interpretation. In `graph_transformer_utils.cc`, `QDQSelectorActionTransformer` is registered before `ConvActivationFusion` at Level 2. Therefore, ORT first attempts QDQ-to-QLinear conversion. If that fails for the Conv+Relu pattern, the later Conv activation fusion pass can convert the remaining `Conv + Relu` into `FusedConv`.

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/graph_transformer_utils.cc

The ONNX Runtime unit tests verify this behavior directly. In `qdq_transformer_test.cc`, the `ConvRelu` test expects:

- If the output Q zero point makes `Relu` redundant: `QLinearConv = 1`, `Relu = 0`
- Otherwise: `QLinearConv = 0`, `Conv = 0`, `Relu = 0`, `com.microsoft.FusedConv = 1`

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/optimizer/qdq_transformer_test.cc

The resulting flow is:

```text
Relu redundant:
DQ -> Conv -> Relu -> Q
=> QDQ Conv group selected
=> QLinearConv
=> Relu removed

Relu not redundant:
DQ -> Conv -> Relu -> Q
=> QDQ Conv group selection fails
=> remaining float Conv + Relu is selected by ConvActivationFusion
=> com.microsoft.FusedConv remains between surrounding QDQ nodes
```

This means ORT does not generally handle this case as `QLinearConv` plus a separate `Relu`. For this pattern, `Relu` must be removable as a redundant quantization clip for QDQ Conv conversion to proceed. Otherwise, under the usual CPU Level 2 optimization path, the surviving `Conv + Relu` is handled by `ConvActivationFusion` as `FusedConv`.

#### 4.5 Conservative interpretation of U8S8 on the ONNX Runtime CPU path

ONNX Runtime's official quantization documentation describes the 8-bit CPU formats as U8U8, U8S8, and S8S8. It also states that S8S8 with QDQ is the default CPU setting and should be the first choice because it balances performance and accuracy. Therefore, the point of using `UINT8` activations with `INT8` weights here is not that U8S8 is generally better than S8S8.

Source:

- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

The same documentation explains why U8S8 is relevant on some x86-64 CPUs. For AVX2 and AVX512 x86 paths that use `VPMADDUBSW`, ORT uses that instruction for U8S8 performance. `VPMADDUBSW` multiplies unsigned bytes by signed bytes and accumulates into 16-bit intermediate values, so saturation can occur when the intermediate result does not fit. ORT notes that this is usually not a major issue for the final result, but if accuracy drops significantly, `reduce_range` or U8U8 can be tested. The documentation also notes that this saturation issue does not apply to x64 with VNNI or to Arm.

Source:

- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8

The ONNX Runtime source code is consistent with that explanation. The CPU `MatMulInteger` path passes A/B signedness into MLAS QGEMM, and MLAS selects the dispatch based on that pair. On x86/x64 builds, `AIsSigned == false` and `BIsSigned == true` maps to the U8S8 dispatch path. Platform initialization then selects a CPU-specific implementation when the build and hardware support it.

Source:

- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/quantization/matmul_integer.cc
- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm.h
- https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/platform.cpp

For this machine, CPUID reports AVX2, AVX512F, and AVX512VNNI support, so the x86 CPU discussion is relevant. Because VNNI is available, the exact behavior should not be inferred from the AVX2 `VPMADDUBSW` path alone; it depends on the installed ONNX Runtime build and runtime feature checks. In this document, U8S8 should therefore be interpreted as a plausible CPU-oriented rewrite target that must be validated with saved profiles and output comparisons, not as a blanket optimization rule.
