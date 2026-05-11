# Docs Overview

This folder contains the working documentation for the AI PC performance lab. It explains how the test environment is configured, which tools are used for capture and profiling, and how model-specific test procedures are organized. Finally, it pairs benchmark results with clear explanations and "why" focused code to help beginners and practitioners understand performance behavior.


## Top-Level Files

- `00_setup.md`: Machine setup record, installed software versions, driver details, validation notes, and known setup issues.
- `01_tools.md`: Repository tooling for environment capture and command execution capture.
- `02_profiling_tools.md`: External profiling and telemetry tools used for tracing, monitoring, and analysis.

## Folders

### `CNNs`

This folder contains CNN-specific test plans, observations, and precision notes.

- `01_cnn_smoke_test.md`: End-to-end CNN smoke test procedure for ResNet50 across CPU, iGPU, and NPU, including model preparation, execution steps, and observed results.
- `02_power_test.md`: power test procedure for comparing latency and software-visible power telemetry across CPU, iGPU, and NPU.
- `03_ONNX_optimization.md`: ONNX optimization notes for CNN inference, including model export, graph rewriting considering the runtime low-level optimization.  

### `LLMs`

TBD. This folder is expected to contain LLM-specific smoke tests, profiling notes, and power or accuracy studies once that work is added to the repository.

## How To Use This Folder

If you are starting a new test flow, read the documents in this order:

1. `00_setup.md`
2. `01_tools.md`
3. `02_profiling_tools.md`
4. The model-family folder you are working in, such as `CNNs`
