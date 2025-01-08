# LLaMA.cpp HTTP Server Example Configuration

## Overview

The `llama.cpp` HTTP server allows you to interact with large language models through a REST API. This document provides examples for configuring and running the server, focusing on embeddings, completions, and batch processing.

## Quick Start

### 1. Install Dependencies

Ensure your system meets the requirements for building and running `llama.cpp`. Use the appropriate CMake configuration:

```bash
# For Vulkan support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=1
# For CUDA support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=1
```

### 2. Start the Server

Run the server with your desired configuration:

```bash
./build/bin/llama-server -m models/7B/ggml-model.gguf --ctx-size 2048 --ubatch-size 512
```
> **Note**: Adjust `--ubatch-size` (default: 512) to match your hardware capacity and token processing needs.

### 3. Verify Server Health

Check if the server is running:

```bash
curl http://127.0.0.1:8080/health
```

## Batch Size and Token Limits

### What is `--ubatch-size`?

The `--ubatch-size` flag determines the **maximum number of tokens** processed per batch. A larger batch size can improve throughput but may require more memory.

#### Default Values

- `--ubatch-size`: 512
- `--ctx-size`: 2048 (maximum context length)

#### Best Practices

- Set `--ubatch-size` based on available memory:
  - **Small Models**: Use 512-1024.
  - **Large Models**: Use 128-512 to avoid memory issues.
- Ensure your batch size aligns with client settings (e.g., `--batch-size` in CLI).

## Embeddings API Example

### Generate Embeddings with Custom Batch Size

```bash
./build/bin/llama-server -m models/7B/ggml-model.gguf --ctx-size 2048 --ubatch-size 256
```

On the client side, configure `--batch-size` to match:

```bash
python -m llama_cpp_client.embedding --batch-size 256 --filepath data.txt
```

### Debugging Tip

If you encounter `500 Internal Server Error`:

1. Check the server logs for `input is too large to process`.
2. Reduce `--batch-size` or `chunk-size` in the client.

## Debugging and Troubleshooting

### Common Errors

- **Input Overflow**: Caused by exceeding `--ubatch-size`.
- **Mismatched Tokenization**: Ensure `--chunk-size` includes special tokens.

### Debugging Commands

- Enable verbose logging on the server:
  ```bash
  ./build/bin/llama-server --verbose
  ```
- Inspect client logs with `--verbose`:
  ```bash
  python -m llama_cpp_client.embedding --verbose --batch-size 256
  ```
