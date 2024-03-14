# llama.cpp client

## Description

The `llama.cpp` client is a experimental front-end web app for interacting with
`llama.cpp`, a powerful tool for natural language processing and text
generation. This client enables seamless communication with the `llama.cpp`
server, making it easy to integrate and interact with `llama.cpp`'s
capabilities.

## Features

- [ ] Interact with the `llama.cpp` server using a simple web ui.
- [ ] Connect to `llama.cpp` server for text generation and conversation.
- [ ] Utilize predefined grammars for precise text generation.
- [ ] Define custom grammars to guide model behavior.
- [ ] Access function schemas for enabling function calls during interactions.

## Getting Started

To get started with the `llama.cpp` client, follow these steps:

1. **Clone the repositories**: Use Git to clone both the `llama-cpp-client` and
   `llama.cpp` repositories onto your local machine or server.

   ```sh
   git clone https://github.com/teleprint-me/llama-cpp-client
   cd llama-cpp-client
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   ```

   Note that `git` will ignore the `llama.cpp` repository.

2. **Build and install `llama.cpp`**: Use the provided instructions to build and
   install `llama.cpp`. For example, you can use CMake to build the library with
   CUBLAS support.

   ```sh
   make LLAMA_CUBLAS=1  # Alternatively, you can use LLAMA_VULKAN=1
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

4. **Access the web UI**: Open your preferred web browser and visit
   `localhost:8080` to access the `llama.cpp` client's web UI. From here, you
   can interact with the `llama.cpp` server for text generation and
   conversation.

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
