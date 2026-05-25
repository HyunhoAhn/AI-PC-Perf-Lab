# LLMs on AI PCs: Overview

## 0. Purpose

This document gives a high-level overview of how Large Language Models, or LLMs, run on an AI PC.

The goal is not to provide step-by-step setup instructions, benchmark commands, benchmark metrics, or performance results. Those should be covered in separate documents. Instead, this page explains the basic execution structure behind local LLM inference:

```text
model preparation
  -> quantization / optimization
    -> model format
      -> runtime
        -> backend / execution provider
          -> CPU / iGPU / NPU / Hybrid
```

This structure is important because an LLM does not automatically run on every accelerator in the system. Whether the model uses the CPU, integrated GPU, or NPU depends on the model format, runtime, backend, driver stack, and hardware support.

In this repository, the `LLMs` folder is intended to contain LLM-specific notes, experiments, and future smoke tests. This document serves as the conceptual entry point for that work.

---

## 1. AI PC hardware for LLMs

Modern AI PCs usually contain three relevant compute blocks:

| Component | Typical role in local LLM workloads |
|---|---|
| CPU | Universal fallback path, tokenizer work, orchestration, smaller models, and unsupported operators |
| iGPU | High-throughput local inference path, especially for GGUF / llama.cpp-style workloads |
| NPU | Power-efficient AI accelerator, usually accessed through a vendor runtime, ONNX Runtime execution provider, Windows ML, or a higher-level framework |

A key point is that these devices are not selected only by hardware capability. They are selected through software.

For example, AMD's Ryzen AI LLM documentation separates execution modes into OGA-based NPU-only, OGA-based hybrid NPU+iGPU, GPU execution through llama.cpp, and CPU fallback paths. See: [AMD Ryzen AI LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html).

This means that "running an LLM on an AI PC" is really a software stack question:

```text
Which model format?
Which quantization or optimization method?
Which runtime?
Which backend or execution provider?
Which hardware device does that backend actually target?
```

---

## 2. Memory model: system RAM, shared memory, and iGPU memory

AI PCs with integrated GPUs often use a unified physical memory pool. The CPU and iGPU are not connected to separate physical DRAM pools like a desktop CPU plus discrete GPU system. Instead, memory is carved up and exposed differently to the OS and applications.

For LLMs, the important distinction is:

| Term | Meaning |
|---|---|
| System RAM | Memory available to the CPU and general OS processes |
| Shared graphics memory | System memory that the GPU may access as graphics memory |
| Dedicated graphics memory / VGM | A portion of system RAM reserved and exposed as GPU-dedicated memory |
| Total graphics addressable memory | Dedicated graphics memory plus shared graphics memory |

AMD's Variable Graphics Memory, or VGM, converts part of system RAM into OS-visible dedicated graphics memory for the iGPU. AMD notes that changing VGM requires a restart, reduces RAM available to the CPU, and creates a contiguous dedicated memory block for iGPU-heavy AI workloads. See: [AMD VGM FAQ](https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html).

This matters for LLMs because many local LLM applications are designed around the idea of "VRAM capacity." On an integrated GPU system, VGM can make the iGPU look more like a large-VRAM device to software that expects dedicated graphics memory.

However, this should not be confused with NPU memory. Increasing iGPU VGM does not automatically mean that an NPU runtime can use that memory in the same way. The iGPU and NPU execution paths are usually exposed through different runtimes, drivers, and backend mechanisms.

---

## 3. Model preparation

Before an LLM can run locally, it usually needs to be prepared for the target runtime.

At a high level, model preparation can include:

| Step | Meaning |
|---|---|
| Downloading model weights | Getting the original model checkpoint from a model hub or vendor source |
| Converting format | Changing the model into the format required by the runtime |
| Quantizing weights | Reducing precision to lower memory usage and sometimes improve speed |
| Optimizing graph/runtime artifacts | Rewriting or compiling the model for a backend, EP, or device target |
| Packaging metadata | Storing tokenizer, architecture, context, chat template, and runtime-specific information |
| Selecting runtime variant | Choosing the artifact that matches CPU, iGPU/GPU, NPU, or hybrid execution |

The key idea is that a model name alone is not enough.

For example, "Llama 8B" or "Qwen 7B" does not fully describe what can run on the machine. The actual deployable artifact might be:

```text
Llama 8B FP16 safetensors
Llama 8B GGUF Q4_K_M
Llama 8B GPTQ INT4
Llama 8B AWQ INT4
Llama 8B ONNX INT4
Llama 8B Ryzen AI pre-optimized OGA model
Foundry Local catalog model variant optimized for CPU / GPU / NPU
```

These are not equivalent from a runtime perspective. They may target different runtimes, use different quantization schemes, and expose different acceleration paths.

---

## 4. Quantization

Quantization is one of the most important concepts for local LLMs.

LLMs are large because they contain billions of parameters. Each parameter is normally stored as a numerical value. Reducing the precision of those values reduces the memory footprint.

A simplified way to think about it is:

```text
higher precision  -> larger memory footprint, usually better quality
lower precision   -> smaller memory footprint, often faster, possible quality loss
```

Common precision levels include:

| Precision / quantization | Rough meaning |
|---|---|
| FP32 | 32-bit floating point weights; usually too large for local LLM inference |
| FP16 / BF16 | 16-bit floating point weights; common high-quality inference format |
| INT8 / Q8 | 8-bit quantized weights or tensors |
| INT4 / Q4 | 4-bit quantized weights or tensors |
| Mixed quantization | Different tensors, weights, or operations use different precision levels |

For GGUF-based workflows, llama.cpp provides a quantization tool that converts higher-precision GGUF files, such as F32 or BF16, into smaller quantized formats. Its documentation notes that quantization reduces model size and can speed up inference, but may introduce accuracy loss. See: [llama.cpp quantization documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

For ONNX Runtime GenAI / Ryzen AI style workflows, quantization may involve different tools and formats. AMD Quark, for example, documents UINT4 AWQ quantization and export to ONNX for ONNX Runtime GenAI. See: [AMD Quark UINT4 OGA tutorial](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/tutorial_uint4_oga.html).

### 4.1 Quantization is not only "4-bit vs 8-bit"

In LLMs, quantization has several separate axes:

| Axis | Example |
|---|---|
| Weight storage precision | FP16, INT8, INT4, NF4, GGUF Q4_K_M |
| Activation precision | FP16/BF16 activations, INT8 activations, mixed precision activations |
| Compute precision | INT4 weights with FP16/BF16 accumulation or compute |
| KV cache precision | FP16 KV cache, quantized KV cache, runtime-specific cache type |
| Algorithm | GPTQ, AWQ, SmoothQuant, bitsandbytes NF4, GGUF K-quants, importance-matrix quantization |
| Runtime packaging | GGUF, ONNX, safetensors plus quantization config, vendor-optimized package |

This is why labels like `Q4`, `INT4`, `NF4`, `GPTQ`, and `AWQ` should not be treated as interchangeable. They may all involve low-bit weights, but they describe different storage formats, quantization algorithms, runtime assumptions, and hardware kernels.

### 4.2 Weight-only vs weight-and-activation quantization

Many LLM inference paths use weight-only quantization.

A common pattern is:

```text
weights:      4-bit or 8-bit
activations:  FP16 or BF16
compute:      runtime/backend dependent
```

This is often written in shorthand as:

```text
W4A16 = 4-bit weights, 16-bit activations
W8A16 = 8-bit weights, 16-bit activations
W8A8  = 8-bit weights, 8-bit activations
```

Weight-only quantization is popular because weights dominate model storage, while leaving activations in higher precision can avoid some of the quality loss that happens with naive activation quantization.

Activation quantization is harder for LLMs because activation distributions can contain outliers. The LLM.int8() paper showed that large transformer models can have extreme outlier features, and bitsandbytes' LLM.int8() implementation keeps outlier-related computation in 16-bit while using 8-bit matrix multiplication for the remaining values. See: [LLM.int8 paper](https://arxiv.org/abs/2208.07339) and [bitsandbytes Linear8bit documentation](https://huggingface.co/docs/bitsandbytes/reference/nn/linear8bit).

This is one reason LLM quantization often uses mixed precision rather than making every tensor and every operation the same low-bit type.

### 4.3 GGUF quantization names

GGUF files often contain names like:

```text
Q4_0
Q4_1
Q4_K_S
Q4_K_M
Q5_K_M
Q6_K
Q8_0
IQ4_NL
IQ3_XS
```

These names are compact labels for llama.cpp / GGML quantization types. They should be read as format-specific names, not as generic mathematical precision labels.

A rough guide:

| Example | Rough interpretation |
|---|---|
| `Q4_0`, `Q4_1` | Older 4-bit GGML/GGUF quantization families |
| `Q8_0` | 8-bit legacy quantization; much larger, usually close to original quality |
| `Q4_K_M` | 4-bit K-quant, medium preset; common size/quality compromise |
| `Q5_K_M` | 5-bit K-quant, medium preset; larger than Q4_K_M, usually less quality loss |
| `Q6_K` | 6-bit K-quant; larger again, usually closer to FP16 quality |
| `IQ*` | Importance-matrix-aware quantization family |

In labels such as `Q4_K_M`:

```text
Q      = quantized
4      = nominal low-bit weight level
K      = K-quant family used by llama.cpp / GGML
M      = medium preset in a small / medium / large family
```

The suffix does not always mean that every tensor is stored with exactly that many bits. K-quant and importance-matrix quantization can use mixed tensor types internally. The llama.cpp quantizer exposes options such as `--pure`, `--tensor-type`, `--output-tensor-type`, and `--token-embedding-type`, which shows that different tensors can use different quantization types inside one GGUF artifact. See: [llama.cpp quantization options](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

Hugging Face's GGUF documentation lists several GGUF quantization types and their approximate bits-per-weight. For example, it describes `Q4_K` as a 4-bit quantization type with super-block structure that results in about 4.5 bits per weight, and `Q5_K` as resulting in about 5.5 bits per weight. See: [Hugging Face GGUF quantization types](https://huggingface.co/docs/hub/en/gguf#quantization-types).

Practical interpretation:

```text
Q8_0 / Q6_K  -> larger, usually higher quality
Q5_K_M       -> larger than Q4, often good quality
Q4_K_M       -> common balance point for local inference
Q3 / Q2 / IQ -> smaller, but quality and backend behavior need more care
```

This is a rule of thumb, not a benchmark result. The best choice depends on model family, model size, backend, memory budget, and workload.

### 4.4 NF4 and bitsandbytes-style 4-bit quantization

NF4 means NormalFloat4. It is a 4-bit data type introduced in the QLoRA work and commonly used through bitsandbytes-based Hugging Face workflows.

NF4 is different from GGUF `Q4_K_M`.

A useful distinction is:

| Term | Typical context |
|---|---|
| `GGUF Q4_K_M` | llama.cpp / GGUF local inference artifact |
| `NF4` | bitsandbytes / QLoRA-style 4-bit quantization for normally distributed weights |
| `FP4` | Another 4-bit floating-point-style representation used in some 4-bit workflows |
| `INT4` | Integer 4-bit quantization; exact meaning depends on runtime and packing scheme |

Hugging Face's bitsandbytes documentation describes NF4 as a 4-bit data type for normally distributed data and notes that QLoRA-style 4-bit models can choose compute data types such as BF16 separately from the 4-bit storage type. See: [bitsandbytes 4-bit documentation](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit) and [Transformers bitsandbytes quantization docs](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

The practical takeaway is:

```text
NF4 is a quantization data type.
GGUF is a model file format.
Q4_K_M is a GGUF / llama.cpp quantization preset.
GPTQ and AWQ are quantization algorithms or model-preparation methods.
```

These labels can appear together in the broader ecosystem, but they are not the same layer of the stack.

### 4.5 GPTQ and AWQ

Two common post-training quantization methods for LLMs are GPTQ and AWQ.

| Method | Rough idea | Common use |
|---|---|---|
| GPTQ | One-shot post-training weight quantization using approximate second-order information | Low-bit GPU inference artifacts, often 3-bit or 4-bit weights |
| AWQ | Activation-aware weight quantization; uses activation information to protect important weights | Hardware-friendly low-bit weight-only quantization, including INT4-style deployment paths |

GPTQ was proposed as a one-shot weight quantization method for large generative transformers, reducing weights to low bit widths such as 3 or 4 bits while trying to preserve accuracy. See: [GPTQ paper](https://arxiv.org/abs/2210.17323).

AWQ stands for Activation-aware Weight Quantization. The AWQ paper describes it as a hardware-friendly low-bit weight-only quantization method for LLMs. See: [AWQ paper](https://arxiv.org/abs/2306.00978).

The word "activation-aware" is important. It reflects the idea that not all weights contribute equally under real input activations. Calibration data can be used to identify weights that are more important for preserving model behavior, so the quantizer can reduce quality loss compared with naive rounding.

In local LLM practice, this means a model repository may offer several different artifacts for the same base model:

```text
model-fp16.safetensors
model-GGUF-Q4_K_M.gguf
model-GPTQ-4bit
model-AWQ-4bit
model-ONNX-INT4-AWQ
```

These artifacts may require different runtimes and are not automatically interchangeable.

### 4.6 Why quantization matters for AI PCs

Quantization affects more than model size.

| Area | Effect |
|---|---|
| Memory usage | Determines whether the model can fit in system RAM, iGPU memory, or runtime-managed memory |
| Runtime compatibility | Some runtimes expect specific quantized formats |
| Hardware acceleration | Some backends support only certain precision/operator combinations |
| Output quality | More aggressive quantization can reduce quality |
| Portability | A quantized model for one runtime may not work in another runtime |
| Backend behavior | A quantization type that is efficient on one backend may be slower or unsupported on another |

---

## 5. Model formats

A local LLM workflow usually starts with a model format. The model format determines which runtimes can load the model and which acceleration paths are available.

### 5.1 Hugging Face / safetensors checkpoints

Many models are distributed as Hugging Face repositories containing `safetensors`, tokenizer files, configuration files, and generation metadata.

This is often the source format, not necessarily the final runtime format for AI PC inference.

Typical use:

```text
Hugging Face checkpoint
  -> convert / quantize / optimize
    -> GGUF, ONNX, GPTQ, AWQ, or runtime-specific model package
```

### 5.2 GGUF

GGUF is the common model format used by llama.cpp and many local LLM applications built around it. Hugging Face describes GGUF as a binary format optimized for quick loading and saving, designed for GGML and related executors, and commonly used with llama.cpp. See: [Hugging Face GGUF documentation](https://huggingface.co/docs/hub/en/gguf).

Typical GGUF path:

```text
Hugging Face checkpoint
  -> GGUF conversion
    -> GGUF quantization
      -> llama.cpp / LM Studio / Ollama-style runtime
```

GGUF is especially relevant for AI PCs because it is widely used by local inference tools and supports many quantization variants.

### 5.3 ONNX

ONNX is a graph format commonly used for deployment through ONNX Runtime and related stacks.

ONNX Runtime uses Execution Providers, or EPs, to map supported parts of a model graph to hardware-specific acceleration libraries. The ONNX Runtime documentation describes EPs as the mechanism that allows ONNX models to execute across hardware platforms such as CPU, GPU, FPGA, or specialized NPUs. See: [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/).

Typical ONNX path:

```text
PyTorch / Hugging Face model
  -> ONNX export
    -> optional quantization / optimization
      -> ONNX Runtime / ONNX Runtime GenAI / Windows ML / vendor EP
```

ONNX is especially important for NPU-oriented workflows because many NPU software stacks expose acceleration through ONNX Runtime EPs.

### 5.4 Runtime-specific optimized models

Some workflows rely on pre-optimized model packages rather than arbitrary user-converted models.

For example, AMD's Ryzen AI documentation describes pre-optimized LLMs for hybrid and NPU-only execution through the Ryzen AI software stack. See: [AMD Ryzen AI LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html).

Foundry Local also uses curated hardware-optimized model variants. Its architecture documentation describes a catalog that provides pre-compiled ONNX models tuned for CPU, GPU, and NPU configurations. See: [Foundry Local architecture overview](https://learn.microsoft.com/en-us/azure/foundry-local/concepts/foundry-local-architecture).

Typical path:

```text
vendor-provided or catalog-provided optimized model
  -> matching runtime version
    -> target EP / backend
      -> CPU, GPU/iGPU, NPU, or hybrid execution
```

This path can be less flexible than GGUF, but it may expose acceleration paths that generic model files do not.

---

## 6. Runtime layer

The runtime is the software that loads the model, manages inference, handles token generation, and talks to the backend.

Different runtimes make different assumptions about model format and hardware.

| Runtime / framework | Common model format | Typical role |
|---|---|---|
| llama.cpp | GGUF | Low-level local LLM runtime, commonly used for CPU and GPU/iGPU execution |
| ONNX Runtime / ONNX Runtime GenAI | ONNX | Runtime for ONNX model execution and generative AI workflows |
| Windows ML | ONNX | Windows-managed local inference framework built on ONNX Runtime |
| Lemonade | GGUF, ONNX, and runtime-managed models | Higher-level local AI server/runtime with CPU, GPU, and NPU-oriented paths |
| LM Studio | GGUF on Windows/Linux/PC via llama.cpp; MLX on Apple Silicon | GUI application for local model download, chat, and server-style usage |
| Ollama | Ollama model library / GGUF-derived workflows | Local model manager and API server |
| Foundry Local | ONNX / curated optimized model variants | Application-oriented local AI runtime and SDK, using ONNX Runtime and Windows ML integration on Windows |

Windows ML is a useful example of the runtime layer becoming system-managed. Microsoft describes Windows ML as a Windows local AI inference framework powered by ONNX Runtime, with acceleration through NPU, GPU, and CPU execution providers that Windows manages and keeps up to date. See: [Windows ML overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview).

Foundry Local sits at an even higher level. Microsoft describes it as a local AI solution that handles model acquisition, hardware acceleration, model management, and inference through an SDK and curated model catalog. See: [Foundry Local overview](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local).

Foundry Local should be understood as an ONNX Runtime / ONNX Runtime GenAI based application runtime rather than a GGUF runtime. Its architecture documentation says the Core API calls into ONNX Runtime for model execution, and on Windows it integrates with WinML for execution provider registration. The ONNX Runtime GenAI repository also states that ONNX Runtime GenAI powers Foundry Local, Windows ML, and the Visual Studio Code AI Toolkit. See: [Foundry Local architecture overview](https://learn.microsoft.com/en-us/azure/foundry-local/concepts/foundry-local-architecture) and [ONNX Runtime GenAI README](https://github.com/microsoft/onnxruntime-genai).

On Ryzen AI systems, Foundry Local can be a Ryzen AI NPU path when a compatible model variant, driver, and execution provider are available. AMD's Ryzen AI Windows ML documentation says LLM models can be run on AMD NPU using Foundry Local or Windows ML APIs, and Microsoft lists VitisAI as an AMD NPU execution provider in the Windows ML EP catalog. See: [AMD Ryzen AI model deployment with Windows ML](https://ryzenai.docs.amd.com/en/latest/winml/model_deployment.html) and [Windows ML execution providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers).

LM Studio should be understood as GGUF-centered on PC, not as an ONNX/NPU-first runtime. The LM Studio docs state that LM Studio supports running LLMs on Mac, Windows, and Linux using llama.cpp, and also supports MLX on Apple Silicon Macs. See: [LM Studio documentation](https://lmstudio.ai/docs/app).

Lemonade is also a higher-level path. Its README describes it as a local AI server that can serve models through standard OpenAI, Anthropic, and Ollama APIs, with AMD optimizations for Ryzen AI, Radeon, and Strix Halo PCs. See: [Lemonade README](https://github.com/lemonade-sdk/lemonade).

---

## 7. Backend and execution provider layer

The backend or execution provider is the part of the stack that actually maps computation to hardware.

This is the layer where many misunderstandings happen.

A user-facing application may say:

```text
Run local LLM
```

But internally the path may be:

```text
LM Studio
  -> llama.cpp
    -> Vulkan backend
      -> iGPU
```

or:

```text
Foundry Local
  -> ONNX Runtime GenAI / ONNX Runtime
    -> Windows ML execution provider registration
      -> VitisAIExecutionProvider
        -> AMD Ryzen AI NPU
```

or:

```text
Windows ML
  -> ONNX Runtime
    -> VitisAIExecutionProvider
      -> AMD Ryzen AI NPU
```

or:

```text
Lemonade
  -> OGA / ONNX Runtime GenAI
    -> NPU + iGPU hybrid execution
```

or:

```text
Ollama
  -> GPU backend
    -> AMD Radeon GPU path
```

The backend layer determines whether the model runs on CPU, iGPU, NPU, or a hybrid combination.

ONNX Runtime exposes this concept explicitly through Execution Providers. A provider can claim supported nodes or subgraphs and execute those parts on the target hardware. Unsupported parts may remain on another provider, often CPU. See: [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/).

Windows ML builds on this idea by managing EP discovery, registration, and updates at the Windows level. AMD's Ryzen AI Windows ML documentation describes this as a system-level EP management layer for ONNX Runtime that can target CPU, GPU, or NPU devices, including the VitisAIExecutionProvider for AMD NPUs. See: [Ryzen AI Windows ML Execution Provider documentation](https://ryzenai.docs.amd.com/en/latest/winml/winml_ep.html).

---

## 8. Conceptual execution paths

The local LLM ecosystem on an AI PC can be understood as several broad paths.

### 8.1 GGUF / llama.cpp path

```text
Hugging Face model
  -> GGUF
    -> quantized GGUF, such as Q4_K_M or Q5_K_M
      -> llama.cpp-based runtime
        -> CPU or iGPU/GPU backend
```

This path is flexible and widely used. It is the foundation for many local LLM tools.

Typical frontends include:

```text
llama.cpp directly
LM Studio on Windows/Linux/PC
Ollama
some Lemonade backends
```

### 8.2 ONNX / ONNX Runtime path

```text
PyTorch / Hugging Face model
  -> ONNX
    -> quantization / optimization, such as INT4 or AWQ-style flow
      -> ONNX Runtime / ONNX Runtime GenAI
        -> execution provider
          -> CPU, GPU, NPU or Hybrid
```

This path is more deployment-oriented. It is especially relevant when the goal is to use hardware-specific EPs or a managed inference stack.

Typical frontends include:

```text
ONNX Runtime
ONNX Runtime GenAI
Windows ML
Ryzen AI OGA flow
Foundry Local runtime layer
```

### 8.3 Windows ML / Foundry Local path

```text
Foundry Local catalog model or compiled ONNX model
  -> Foundry Local Core API
    -> ONNX Runtime / ONNX Runtime GenAI
      -> Windows ML EP management on Windows
        -> CPU, GPU, NPU, or vendor-specific EP
```

This path is designed for application integration. It hides much of the hardware selection and model acquisition logic behind an SDK and model catalog.

On Windows, Foundry Local integrates with Windows ML for execution provider registration. On Ryzen AI PCs, the relevant NPU execution provider is the AMD VitisAIExecutionProvider when the model, driver, and EP support line up.

This should not be described as "any Hugging Face model automatically runs on the NPU." A safer description is:

```text
Foundry Local can use NPU acceleration through Windows ML / ONNX Runtime EPs
when a compatible model variant and execution provider are available.
```

### 8.4 Vendor-optimized NPU / hybrid path

```text
pre-optimized model
  -> matching runtime
    -> NPU or NPU+iGPU backend
      -> AI PC accelerator path
```

This path is less generic but can expose acceleration modes that are not available to arbitrary model files.

AMD's Ryzen AI documentation describes NPU-only and hybrid NPU+iGPU execution modes through ONNX Runtime GenAI, while GPU-only acceleration is enabled through llama.cpp. See: [AMD Ryzen AI LLM overview](https://ryzenai.docs.amd.com/en/latest/llm/overview.html).

### 8.5 Application-friendly local server path

```text
model catalog or imported model
  -> local runtime/server
    -> backend selected by runtime
      -> CPU, iGPU, GPU, NPU, or hybrid
```

This path is designed for ease of use and application integration.

Examples include:

```text
Lemonade
Ollama
LM Studio server mode
Foundry Local optional REST server
```

The advantage is that applications can connect through familiar APIs. The tradeoff is that the actual backend selection can be hidden behind the tool, so later documents should verify which hardware path is actually being used.

---

## 9. Why the stack matters

The same physical AI PC can behave very differently depending on the stack.

For example:

```text
Same hardware
Same model family
Different quantization
Different model format
Different runtime
Different backend
Different device used
```

A GGUF model may run well through a llama.cpp Vulkan backend on the iGPU, while an ONNX model may target an NPU through an ONNX Runtime EP. A higher-level application may hide this distinction, but the distinction still determines memory usage, compatibility, and acceleration behavior. Furthermore, llama.cpp also can ROCm backend based, not Vlukan backend based.   

Therefore, useful naming style is:

```text
model family / parameter size
+ quantization method or precision
+ model format
+ runtime
+ backend or EP
+ target device
```

Example:

```text
Qwen 7B
+ INT4 AWQ
+ ONNX
+ ONNX Runtime GenAI
+ VitisAIExecutionProvider
+ NPU
```

or:

```text
Llama 8B
+ Q4_K_M
+ GGUF
+ llama.cpp
+ Vulkan
+ iGPU
```

or:

```text
Phi model variant
+ Foundry Local catalog optimized model
+ ONNX
+ Foundry Local / ONNX Runtime GenAI
+ Windows ML-managed EP
+ CPU/GPU/NPU depending on selected variant
```

---

## 11. Summary

Running an LLM on an AI PC is not just a question of whether the machine has a CPU, iGPU, or NPU.

It is a stack question:

```text
model preparation
  -> quantization / optimization
    -> model format
      -> runtime
        -> backend / execution provider
          -> hardware device
```

The most important practical distinction is:

| Path | Typical format | Typical hardware direction |
|---|---|---|
| GGUF / llama.cpp | GGUF with Q/IQ quantization variants | CPU or iGPU/GPU |
| LM Studio | GGUF via llama.cpp on PC; MLX on Apple Silicon | CPU or GPU/iGPU depending on runtime backend |
| ONNX / ONNX Runtime | ONNX with runtime-specific quantization | CPU, GPU, or NPU through EPs |
| Ryzen AI OGA | optimized ONNX / OGA models | NPU-only or NPU+iGPU hybrid |
| Windows ML | ONNX | Windows-managed CPU/GPU/NPU path |
| Foundry Local | ONNX / curated optimized variants | ONNX Runtime / OGA-based runtime with Windows ML EP integration on Windows; can target NPU when compatible |
| Lemonade / Ollama | Tool-managed | Depends on selected backend and model format |

