# llama.cpp client

## Description

The `llama.cpp` client is a experimental front-end client library for interacting with `llama.cpp`, a powerful tool for natural language processing and text generation. This client enables seamless communication with the `llama.cpp` server, making it easy to integrate and interact with `llama.cpp`'s capabilities.

## Features

**Overview**  
`llama-cpp-client` explores the capabilities of `llama.cpp` through various tools and interfaces, focusing on interaction, experimentation, and extensibility.

### Roadmap

#### Interaction with `llama.cpp`
- Simple API for handling requests and responses.
- Command-line interface (CLI) for interacting with the server.
- Text-based UI using `rich` for enhanced visual feedback.
- Web-based UI for browser-based interaction.
- Robust chat completions with multi-turn conversation support.
- Backus-Naur Form (BNF) grammar integration for structured output.

#### Integration with `llama.cpp`
- Text embeddings with SQLite storage for retrieval and similarity searches.
- Fully functional text completions (single and streaming outputs).
- Function schemas defining callable APIs during interactions.
- Function calls for dynamic actions based on model outputs.

#### Planned Enhancements
- Basic text editor for prompts, datasets, and code:
  - Compose and edit prompts or datasets.
  - Run and visualize completions and embeddings.
  - Manage and explore history or embeddings.
  - Code completion, navigation, formatting, debugging, tracing, testing, and documentation.

---

**Note**  
All features are work in progress, with active development focused on refining and implementing the roadmap.

**Status**  
`llama-cpp-client` is in its early stages, with ongoing efforts to expand and polish functionality.

## Getting Started

To get started with the `llama.cpp` client, follow these steps:

1. **Clone the repositories**: Use Git to clone both the `llama-cpp-client` and
   `llama.cpp` repositories onto your local machine or server.

   ```sh
   git clone https://github.com/teleprint-me/llama-cpp-client
   cd llama-cpp-client
   ```

   Note that `git` will ignore the `llama.cpp` repository.

   ```sh
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```

2. **Build and Install Llama.Cpp**

   To build and install `llama.cpp`, use the following instructions:

   - Vulkan Support

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=1 -DGGML_VULKAN_DEBUG=0 -DGGML_CCACHE=0
   ```

   - CUDA Support

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=1 -DGGML_CUDA_DEBUG=0 -DGGML_CCACHE=0
   ```

   - BLAS Support

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=1 -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_CCACHE=0
   ```

   - CPU Support

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=0
   ```

   - Compile with All CPU Cores

   ```bash
   cmake --build build --config Debug -j $(nproc)
   ```

3. **Run the `llama.cpp` server**: Use the provided instructions to run the
   `llama.cpp`
   [server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
   with your chosen model and configuration settings.

   ```sh
   ./llama.cpp/build/bin/llama-server -m [model path here] --ctx-size [int] --n-gpu-layers [int] --path app
   ```

   Note that you can extend the front end by running the server binary with `--path`.

### WebUI

**How to use the web user interface**

Open your preferred web browser and visit `localhost:8080` to access the `llama.cpp` client's web UI. From here, you can interact with the `llama.cpp` server for text generation and conversation.

**Note**: The WebUI is currently a limited prototype for completions.

### CLI

**How to use the command-line interface**

```sh
python -m llama_cpp_client.client -n llama-3-test --stop "<|eot_id|>"
```

### API

**How to use the application programming interface**

```python
from llama_cpp_client.request import LlamaCppRequest

# Initialize the LlamaCppRequest instance
llama_cpp_request = LlamaCppRequest(base_url="http://127.0.0.1", port="8080")

# Define the prompt for the model
llama_prompt = "Once upon a time"

# Prepare data for streaming request
llama_data = {"prompt": llama_prompt, "stream": True}

# Request the models stream generator
llama_generator = llama_cpp_request.stream("/completion", data=llama_data)

# Track completions
completions = []

# Generate the model's response
llama_output = ""
for response in llama_generator:
   if "content" in response:
      token = response["content"]
      llama_output += token
      # Print each token to the user
      print(token, end="")
      sys.stdout.flush()

# Add padding to the model's output
print()

# Append the completion
completions.append({"prompt": llama_prompt, "output": llama_output})
```

Note that most of the Python API modules for `llama_cpp_client` can be executed as a CLI tool providing an example, test, and output sample all in one place.

```sh
python -m llama_cpp_client.request
```

The general idea is to keep the implementation as simple as possible for now. 

Check out the source code for more examples.

#### Summary

By following these steps, you should be able to get started with the `llama.cpp`
client and begin exploring its capabilities. For more detailed documentation and
examples, please refer to the `llama-cpp-client` documentation.

## Documentation

Refer to the [llama.cpp client documentation](/docs) for detailed documentation
and examples.

## License

This project is licensed under the [AGPL License](LICENSE.md).

## Acknowledgments

- The `llama.cpp` team for developing an incredible natural language processing tool.
