# rag-langchain-ryzen Local RAG Verification

## Project Link

- [HyunhoAhn/rag-langchain-ryzen](https://github.com/HyunhoAhn/rag-langchain-ryzen)

## What This Project Tests

`rag-langchain-ryzen` is a small CLI-based local RAG verification project for a Windows 11 Ryzen AI MAX development machine.

The goal is to verify that a minimal local Retrieval-Augmented Generation, or RAG, flow works end to end on the machine:

```text
local documents
  -> document loading and chunking
    -> local embedding generation
      -> local vector index
        -> retrieval
          -> prompt construction
            -> Lemonade Server chat completion
              -> source-grounded answer
```

This is useful as an early LLM smoke test because it checks more than simple token generation. It verifies that the local machine can run a practical application-style LLM workflow where retrieval, embedding, vector storage, prompt assembly, and local answer generation all work together. This project intentially utilizes Korean to test its bilingual capability.

## Main Components

| Component | Role in the test |
|---|---|
| Lemonade Server | Exposes the local LLM through an OpenAI-compatible API endpoint. |
| LangChain | Orchestrates document loading, text splitting, retrieval, prompt construction, and answer generation. |
| Chroma | Stores the persistent local vector index used for retrieval. |
| HuggingFace sentence-transformers | Generates embeddings locally, using CPU by default in this project. |
| Local documents | `.txt`, `.md`, and `.pdf` files used as the RAG knowledge source. |
| CLI commands | Provide repeatable checks for server connectivity, ingestion, retrieval, asking, and retrieval evaluation. |

## What Lemonade Is Used For

In this project, Lemonade is used as the local model serving layer.

The RAG application does not call a hosted cloud LLM. Instead, it sends chat-completion requests to a Lemonade Server endpoint such as:

```text
http://localhost:13305/v1
```

Because Lemonade exposes an OpenAI-compatible API, the application can use a familiar chat model interface while still targeting a local Ryzen AI PC workflow. The configured model name is passed through the `LEMONADE_CHAT_MODEL` environment variable.

This project assumes that Lemonade Server is installed, configured, and already running separately. The project checks whether Lemonade is reachable, but it does not install Lemonade, start the server, configure models, or prove which hardware block is used internally by Lemonade.

## What Is Actually Verified

The project verifies the following application-level behavior:

- Lemonade Server can be reached through the configured OpenAI-compatible base URL.
- The configured chat model can be selected from the application.
- Local `.txt`, `.md`, and `.pdf` files can be ingested.
- Documents can be split into chunks and embedded locally.
- Chunks can be stored in a persistent Chroma collection.
- Retrieval can be inspected directly from the CLI.
- Retrieved chunks can be inserted into a RAG prompt.
- Lemonade can generate an answer using the retrieved context.
- Basic retrieval quality can be checked with Hit@K and MRR.

## What This Project Is Not

This project should not be read as a formal benchmark harness.

It does not publish latency, throughput, memory, power, or accuracy benchmark results. It also does not claim NPU acceleration by itself. Whether the LLM uses CPU, iGPU, NPU, or a hybrid path depends on the Lemonade model, backend, driver stack, and runtime configuration outside this RAG project.

In the context of this repository, `rag-langchain-ryzen` is best treated as a practical LLM/RAG smoke test:

```text
Can this Ryzen AI PC run a local RAG application through Lemonade and LangChain?
```

After this works, separate benchmark or profiling documents can measure latency, token throughput, memory behavior, power telemetry, retrieval quality, and hardware execution path.
