# 02 Power Test

## Concept Primer

This document defines the procedure for the CNN power test. The goal is to compare latency together with software-visible power and telemetry signals while running the same workload on CPU, NPU, and iGPU.

This is not a rail-level electrical measurement. A stricter study would require direct electrical instrumentation or controlled wall-power measurement. In this repository, the first pass uses software telemetry only.

## Scope

- Run ONNX Runtime inference for `ResNet50` in at least `FP32` and `INT8`.
- Compare `CPU`, `NPU`, and `iGPU` execution.
- Disable ONNX Runtime profiling during the measured runs.
- Reuse a prebuilt NPU model cache when measuring NPU runs.
- Increase the repeat count relative to the smoke test, because `ResNet50` is too short-running for stable software-visible power deltas at low iteration counts.
- Execute multiple runs per configuration and compare both latency and power-related telemetry.

## Preconditions

0. Review `docs/00_setup.md`, `docs/01_tools.md`, and `docs/02_profiling_tools.md`.
1. The Windows environment is ready and the required ONNX Runtime execution providers are available.
2. `models/resnet50.onnx` and any derived model variants needed for the test already exist.
3. External monitoring tools are installed and validated:
   - `AMD uProf`
   - `HWiNFO64`
   - `xrt-smi`

## Tool Roles

- `AMD uProf`: primary CPU-side monitor for core-level and package-level counters.
- `HWiNFO64`: supplemental sensor source for package power, temperature, clocks, fan speed, and board-level telemetry when the platform exposes those sensors.
- `xrt-smi`: estimated NPU power and NPU state.

Package-level telemetry should be treated as a proxy. It is useful for comparison, but it should not be interpreted as a fully isolated iGPU or NPU power measurement unless the platform documentation confirms that attribution.

## Test Design

The current plan is to run the same ONNX Runtime inference workload across devices while collecting external telemetry.

1. Capture the environment for the run.
2. Start the external monitoring tools.
3. Run the benchmark with profiling disabled.
4. For NPU, prepare and reuse model cache artifacts before the measured run.
5. Capture telemetry immediately before the run and again during or after the run, depending on tool capability.
6. Repeat each configuration multiple times.
7. Compare latency and telemetry deltas across devices.

## How to Run

### 1. Capture environment metadata

```powershell
$runId = "02_power_test"
python tools/capture_env.py --run-id $runId
```

### 2. Start external monitors

Start the relevant monitors before running inference:

- `AMD uProf` for CPU core and package telemetry
- `HWiNFO64` for supplemental sensor logging
- `xrt-smi` for NPU estimated power when testing the NPU

Store any exported logs under `results/raw/$runId/` or document their original save location clearly.

### 3. Run the benchmark workload

Use the existing CNN smoke test entrypoint as the workload driver, but increase the repeat count for this power-focused run.

Example NPU command:

```powershell
python tools/run_capture.py --run-id $runId -- python src/CNNs/01_cnn_smoke_test.py --model-path .\models\resnet50.onnx --device npu --disable-fallback --input-shape 3x224x224 --batch 1 --warmup 10 --repeat 20000 --vaip-cache-dir results/raw/$runId --vaip-cache-key resnet50_FP32
```

The repeat count is expected to be tuned. The main requirement is that the run duration is long enough to produce a measurable change in the software telemetry.

## Device-Specific Notes

### CPU

- Use `AMD uProf` as the primary source for CPU power.
- Collect both core-level and package-level counters when available.
- Use `HWiNFO64` as a supplemental source for package power, temperature, clock, and fan data.

### NPU

- Use `xrt-smi` for estimated NPU power.
- Use `AMD uProf` package-level counters as a supplemental system-level proxy.
- Use `HWiNFO64` as a supplemental telemetry source.
- Reuse the model cache before measured runs so that compilation and first-run effects do not dominate the results.

### iGPU

- Use `AMD uProf` package-level counters as a supplemental proxy.
- Use `HWiNFO64` as a supplemental telemetry source.

## Planned Outputs

At minimum, each run should preserve:

- `results/raw/$runId/env_history.jsonl`
- `results/raw/$runId/stdout.log`
- `results/raw/$runId/stderr.log`
- `results/raw/$runId/metadata.jsonl`
- exported telemetry logs from `AMD uProf`, `HWiNFO64`, and `xrt-smi` when applicable

## Notes

- Power should be interpreted together with latency. A device that uses more power but finishes substantially faster may still be preferable depending on the objective.
- Because this workload is relatively light, short runs are likely to understate device differences.
- This document is still a draft and the exact per-device command matrix should be finalized after the monitoring workflow is stabilized.
