# 02 Power Test

## Concept Primer

This document defines the procedure for the CNN power test. The goal is to compare latency together with software-visible power and telemetry signals while running the same workload on CPU, NPU, and iGPU.

This is not a rail-level electrical measurement. A stricter study would require direct electrical instrumentation or controlled wall-power measurement. In this repository, the first pass relies on software telemetry only.
**Important:** AMD uProf is the primary measurement source, and the current analysis assumes that CPU package-level power reflects the combined activity of the CPU, NPU, iGPU, and other on-package components. If this assumption does not hold, the test design should be revised.

## Scope

- Run ONNX Runtime inference for `ResNet50` in at least `FP32` and `INT8`.
- Compare `CPU`, `NPU`, and `iGPU` execution.
- Disable ONNX Runtime profiling during the measured runs.
- Reuse a prebuilt NPU model cache when measuring NPU runs.
- Increase the repeat count relative to the smoke test, because `ResNet50` is too short-running for stable software-visible power deltas at low iteration counts.
- Execute multiple runs per configuration and compare both latency and power-related telemetry.

## Preconditions

0. Review `docs/00_setup.md`, `docs/01_tools.md`, `docs/02_profiling_tools.md`, and `docs\CNNs\01_cnn_smoke_test.md`.
1. The Windows environment is ready and the required ONNX Runtime execution providers are available.
2. **`models/resnet50.onnx` and any derived model variants needed for the test already exist.**
3. External monitoring tools are installed and validated:
   - `AMD uProf` : 5.2.431, C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe
   - `xrt-smi` : 1.7.1 (Ryzen AI), C:\Windows\System32\AMD\xrt-smi.exe

## Tool Roles

- `AMD uProf`: primary CPU-side monitor for core-level and package-level counters.
- `xrt-smi`: estimated NPU power and NPU state.
- `HWiNFO64`: supplemental sensor source for package power, temperature, clocks, fan speed, and board-level telemetry when those sensors are exposed by the platform. Its CPU Package Power metric can be used for an APU, but it is excluded from this test because CLI log export requires a Pro license, and the workflow is intended to remain accessible without additional paid tooling.

This power test uses software-visible telemetry as a proxy for power. Although this is not a direct electrical measurement, it remains useful for relative comparison across devices under the same workload. Package-level telemetry should therefore be interpreted as a comparative indicator rather than a validated full-system CPU+NPU+iGPU power measurement.


## How to Run

### 1. Capture environment metadata

```powershell
$runId = "cnn_power_test"
python tools/capture_env.py --run-id $runId
```
### 2. Capture benchmark execution with performance and power profiling.

Manual run:

```powershell
python src/CNNs/02_power_test.py `
  --run-id cnn_power_test `
  --case-name fp32_npu `
  --attempt-index 1 `
  --telemetry-tool xrt `
  --model-path models/resnet50.onnx `
  --device npu `
  --disable-fallback `
  --input-shape 3x224x224 `
  --batch 1 `
  --warmup 10 `
  --repeat 20000 `
  --shared-vaip-cache-dir results/raw/cnn_power_test `
  --vaip-cache-key resnet50_FP32
```

Recommended automated run:

```powershell
& .\tools\02_cnn_power_test.ps1
```

## Observed Results

At minimum, each run should preserve:

- `results/raw/$runId/env_history.jsonl`
- `results/raw/$runId/stdout.log`
- `results/raw/$runId/stderr.log`
- `results/raw/$runId/metadata.jsonl`
- exported telemetry logs from `AMD uProf`, and `xrt-smi` when applicable
- power profile data in results/raw/$runId/power/$profiler_name/case_name/attempt_*/ directory, where `profiler_name` is `amd_uprof` or `xrt_smi`, and `case_name` is a sanitized string derived from the test parameters (e.g. `resnet50_fp32_npu`).


### 1. Power vs Latency Comparison


| Model (Precision) | Metric | CPU | NPU | iGPU |
|:---:|:---:|---:|---:|---:|
| **FP32 Model** | Delta Power (W) | 85 | 17.1 | 102.7 |
| | Latency (ms) | 6.44 | 5.5 | 2.7 |
| **INT8 Model** | Delta Power (W) | 64.9 | 16.9 | 126.2 |
| | Latency (ms) | 22.7 | 2.9 | 2.2 |

In this table, Power refers to the package-level power measurement reported by AMD uProf during each execution mode, and Latency refers to the average inference time reported by ONNX Runtime.
These power values should be interpreted as package-level proxy measurements observed while running each device configuration, not as isolated rail measurements for the CPU, NPU, or iGPU alone.
Package-level power is evaluated as a delta between idle and working states. For each state, the first 10% of samples is discarded, and the mean of the remaining 90% is used for comparison. This reduces the influence of startup transients, including short spikes and intervals before the workload reaches a stable execution state. The sampling interval is 200 ms. Relative to `01_smoke_test`, the larger iteration count used here reduced latency variability in most configurations.

Several observations follow from these results.
Lower latency on the CPU and iGPU is associated with higher package-level power. A plausible interpretation is that higher computational activity, increased data movement, and elevated clock frequencies contribute to the observed power increase.
The INT8 CPU case shows the opposite tradeoff: package-level power is lower than FP32, but latency is substantially higher. As discussed in `01_cnn_smoke_test`, this behavior is consistent with CPU-side bottlenecks introduced by operations such as Transpose and Q-DQ handling.
The CPU FP32 versus INT8 comparison also shows that lower instantaneous power should not be interpreted as higher efficiency by itself. When latency increases substantially, total energy use may still become less favorable despite the lower sampled power level.
NPU execution is associated with lower package-level power than either CPU or iGPU execution in these measurements. This is most visible in the INT8 configuration, which appears to be the best-optimized variant for this model. Despite only an approximately 0.7 millisecond latency difference relative to the iGPU, measured package-level power drops from 126.2 W to 16.9 W. This result reinforces the need to evaluate latency and power together, particularly for resource-constrained devices.


#### NPU Measured with `xrt-smi`
| fp32_npu | int8_npu |
|---|---|
| 3.422 W | 3.408 W |

This table reports NPU power measured with `xrt-smi`. As in the AMD uProf analysis, idle and working states were sampled at 200 ms intervals, the first 10% of samples was discarded, and the mean of the remaining 90% was used for comparison. The difference is small, but the INT8 case again shows slightly lower power under the same measurement procedure.


### 2. Best Observed Configuration by Device
| Metric | CPU (FP32) | NPU (INT8) | iGPU (FP32) |
|:---:|---:|---:|---:|
| **Delta Power (W)** | 85 | 16.9 | 102.7 |
| **Latency (ms)** | 6.44 | 2.9 | 2.7 |

As noted in `01_cnn_smoke_test`, the iGPU may have shown a stronger result under FP16.
This table compares the best observed configuration selected for each device, rather than a like-for-like precision match across all devices.
Within the configurations presented here, the NPU run again shows the lowest package-level power while maintaining competitive latency. The INT8 model in particular shows a substantial improvement in NPU-associated power efficiency. By contrast, the iGPU provides the lowest latency in this comparison, but also the highest package-level power. These results indicate that device selection and model optimization both influence latency and power, and they should therefore be evaluated jointly when selecting an execution strategy.

### Next Steps
- CPU-side bottlenecks remain present in the INT8 model. The next step is to investigate and optimize those bottlenecks.
