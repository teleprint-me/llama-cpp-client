# llama.cpp client

## Description

The `llama.cpp` client is a experimental front-end client library for interacting with `llama.cpp`, a powerful tool for natural language processing and text generation. This client enables seamless communication with the `llama.cpp` server, making it easy to integrate and interact with `llama.cpp`'s capabilities.

## Features

This project aims to explore the capabilities of `llama.cpp` through various tools and interfaces, with a focus on interaction, experimentation, and extensibility. Below is the current roadmap for features, including work-in-progress and planned items:

### Interaction with `llama.cpp`
- [ ] A **simple API** for handling requests and responses, making it easy to send prompts and receive completions.
- [ ] A **command-line interface (CLI)** for interacting with the server via terminal commands, supporting:
  - [ ] Text embeddings
  - [ ] Text completions
  - [ ] Chat completions
  - [ ] Infill completions
  - [ ] Grammar-based completions
- [ ] A **text-based UI** using `rich` for enhanced visual feedback during interactions.
- [ ] A **web-based UI** for interacting with the model through a browser.

### Integration with `llama.cpp`
- [ ] Support for **text embeddings**, with SQLite storage for efficient retrieval and similarity searches.
- [ ] Fully functional **text completions**, providing single and streaming output.
- [ ] Robust **chat completions**, capable of maintaining context across multi-turn conversations.
- [ ] Integration of **Backus-Naur Form grammars**, enabling structured and constrained output generation.

### Tooling
- [ ] Tools to enable **function schemas**, defining callable APIs during interactions.
- [ ] Support for **function calls**, allowing dynamic actions based on model outputs.

### Planned Enhancements
- [ ] A **basic text editor**, providing an integrated environment to:
  - Compose and edit prompts or datasets.
  - Run and visualize completions and embeddings.
  - Manage and explore history or embeddings within the project.
  - Serve as a bridge between the text UI, CLI, and web interface.

**NOTE**: All features are currently a work in progress, with active development focused on exploring and refining each component.

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
