# AI PC Performance Lab (Ryzen AI / NPU-first)

## Why this repo
- Goal: Provide reproducible benchmarking and profiling for on-device AI workloads such as LLMs and CNNs.
- Focus: Start with AMD Ryzen AI (NPU/iGPU/CPU) reference workflows, then expand into deeper optimization techniques. GPU-focused workflows are planned for later.
- Approach: Pair benchmark results with clear explanations and "why" focused code to help beginners and practitioners understand performance behavior.

## Getting Started
1. Read `docs/00_setup.md` for environment setup and repo walkthrough.
2. Read `docs/00_tools.md` to understand the tools for environment capture and command execution capture.
3. Run the CNN smoke test: `docs/cnn_smoke_test.md`
4. Run LLM reference benchmarks: WIP
5. Profiling playbook: WIP
6. Quantization deep dive: WIP
7. GPU kernel deep dive (optional): WIP

## Repo structure
- `docs/`: setup, methodology, benchmark plans, and profiling playbooks
- `tools/`: CLI entry points for capture, parsing, and reporting
- `src/`: reusable logic for toolchain internals
- `results/raw/`: immutable artifacts including environment capture, run commands, results, and configs
- `images/`: documentation images

## Documentation Workstream
Each benchmark/profiling document is a first-class deliverable and includes:
1. A short concept primer
2. A how-to-run procedure
3. Results and analysis with "why"-focused explanations and code

