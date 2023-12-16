# Example Usage of llama.cpp

llama.cpp is a versatile tool for natural language processing and conversation generation. It allows you to interact with various language models using customizable prompts. In this document, we will explore an example of how to use llama.cpp to generate text based on a predefined prompt template.

## Prompt Templates for Llama Models

llama.cpp is highly adaptable to a variety of prompt templates, making it a versatile tool for interacting with different models. Various organizations and projects have developed specific template structures to cater to their unique objectives. Notably, these templates are commonly used with the Llama, Llama 2, Mistral, and Mixtral models. Let's explore an example template that is shared among these models:

```plaintext
<<SYS>>My name is [Assistant Name] and I am a helpful assistant.<</SYS>>
[INST] Hello! My name is [User Name]. What's your name? [/INST]
Hello, my name is [Assistant Name]. Nice to meet you!
[INST] What can you do? [/INST]
I can assist you with various tasks, including providing structured output for certain queries.
[INST] How can you assist me in my programming projects? [/INST]
```

This template, demonstrating a structured AI interaction, highlights llama.cpp's capability to align with different operational contexts and model requirements. It serves as a common foundation for interactions with these models.

In this template:

- `<<SYS>>` is used to indicate system-level information.
- `[INST]` is used to indicate user instructions or interactions.
- You can replace `[Assistant Name]` and `[User Name]` with specific names or placeholders.
- The assistant responds to user instructions within the `[INST]` sections.

Different projects and models may use variations of this template, but it provides a starting point for structuring prompts according to the requirements of the Llama, Llama 2, Mistral, and Mixtral models.

## Using the llama.cpp Server

To utilize the llama.cpp server with a specific model, you can use the `-m` option to specify the model you want to serve. Here's the basic command:

```sh
./vendor/llama.cpp/server -m MODEL_PATH
```

In this command:

- The `-m` option is used to define the path to the model you want to use.

You have the flexibility to choose any model you prefer to serve. For a deeper understanding of available options and configuration, you can refer to the help documentation by using the following command:

```sh
./vendor/llama.cpp/server --help
```

This will provide you with a comprehensive overview of server-related options and settings.

## Using the llama.cpp Server Endpoints

To interact with llama.cpp using a specific model and prompt, you can make an HTTP POST request to the llama.cpp server. Here's an example `curl` command for making such requests:

```sh
curl -X POST http://127.0.0.1:8080/completion \
     -H "Content-Type: application/json" \
     -d @- <<'EOF'
{
  "prompt": "<<SYS>>My name is Mistral and I am a helpful assistant.<</SYS>>\n[INST] Hello! My name is Austin. What's your name? [/INST]\nHello, my name is Mistral. Nice to meet you!\n[INST] What can you do? [/INST]\nI can assist you with various tasks, including providing structured output for certain queries.\n[INST] How can you assist me in my programming projects? [/INST]",
  "max_tokens": 100,  # Optional parameter
  "temperature": 0.7  # Optional parameter
}
EOF
```

In this `curl` command:

- The `-X POST` option specifies that it's an HTTP POST request.
- The `-H "Content-Type: application/json"` option sets the request header to JSON format.
- The `-d @- <<'EOF'` option sends the JSON data as the request body.

You have the flexibility to customize the `prompt`, `max_tokens`, and `temperature` parameters as needed. However, it's crucial to ensure that the `prompt` follows the specific structure required by the chosen model, as discussed earlier.

## Additional Context

llama.cpp is a flexible tool that has evolved to support various models and prompt structures. Different organizations and projects may adopt specific prompt templates based on their needs. The example provided here is just one illustration of how you can use llama.cpp for text generation and conversation.

Feel free to explore different prompt templates and experiment with llama.cpp to achieve your specific goals.

For more details and customization options, you can refer to the llama.cpp server documentation.
