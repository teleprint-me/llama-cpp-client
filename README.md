# llama.cpp client

## Description

The `llama.cpp` client is a experimental front-end client library for interacting with `llama.cpp`, a powerful tool for natural language processing and text generation. This client enables seamless communication with the `llama.cpp` server, making it easy to integrate and interact with `llama.cpp`'s capabilities.

## Features

- [ ] Interact with the `llama.cpp` server using a simple api.
- [ ] Interact with the `llama.cpp` server using a simple cli.
- [ ] Interact with the `llama.cpp` server using a simple web ui.
- [ ] Connect to `llama.cpp` server for text generation and conversation.
- [ ] Utilize predefined grammars for precise text generation.
- [ ] Define custom grammars to guide model behavior.
- [ ] Access function schemas for enabling function calls during interactions.

**NOTE: All interfaces are currently a WIP (work in progress)**

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

2. **Build and install `llama.cpp`**: Use the provided instructions to build and
   install `llama.cpp`. For example, you can use CMake to build the library with ROCm support. I personally prefer Vulkan when using AMD because Vulkan has better support for a wider range of GPU's than ROCm does.

   Build the library with Vulkan support.

   ```sh
   make LLAMA_VULKAN=1
   ```

   Build the library with CUDA support.

   ```sh
   make LLAMA_CUDA=1
   ```

3. **Run the `llama.cpp` server**: Use the provided instructions to run the
   `llama.cpp`
   [server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
   with your chosen model and configuration settings.

   ```sh
   ./llama.cpp/server -m [model path here] --ctx-size [int] --n-gpu-layers [int] --path app
   ```

   Note that you can extend the front end by running the server binary with
   `--path`.

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

# Generate the model's response
content = ""
for response in llama_generator:
   if "content" in response:
      token = response["content"]
      content += token
      # Print each token to the user
      print(token, end="")
      sys.stdout.flush()

# Add padding to the model's output
print()
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

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

- The `llama.cpp` team for developing an incredible natural language processing
  tool.
