# llama.cpp client

## Description

The llama.cpp client is a Python wrapper for interacting with llama.cpp, a powerful tool for natural language processing and conversation generation. This client enables seamless communication with the llama.cpp server, making it easy to integrate llama.cpp's capabilities into your Python applications.

## Features

- Connect to llama.cpp server for text generation and conversation.
- Utilize predefined grammars for precise text generation.
- Define custom grammars to guide model behavior.
- Interact with the llama.cpp server using simple Python functions.
- Access function schemas for enabling function calls during conversations.

## Getting Started

To get started with the llama.cpp client, follow these steps:

1. [Install llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) on your local machine or server.
2. Install the llama.cpp client Python package using pip:

   ```sh
   pip install llama-cpp-client
   ```

3. Create a connection to the llama.cpp server and start generating text or engaging in conversations with the model.

```python
from llama_cpp_client.client import LlamaCppClient

# Create a client instance
client = LlamaCppClient(host="http://127.0.0.1", port="8080")

# Generate text
response = client.generate_text(prompt="Tell me a story about space exploration.")
print(response)

# Engage in a conversation
conversation = ["Hello, how are you?", "I'm good, thanks. How about you?"]
response = client.generate_conversation(conversation)
print(response)
```

## Documentation

For detailed documentation and examples, please refer to the [llama.cpp client documentation](/docs).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- The llama.cpp team for developing an incredible natural language processing tool.
