"""
Module: llama_cpp_client.client
"""
import json
import sys
from typing import Any, Callable, List

from rich import pretty, print

from llama_cpp_client.format import get_prompt_sequence
from llama_cpp_client.grammar import LlamaCppGrammar
from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.types import ChatMessage


class LlamaCppAPI:
    ...  # TODO


class LlamaCppClient:
    ...  # TODO


def get_current_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather in a given location.

    Parameters:
    location (str): The city and state, e.g. San Francisco, CA
    unit (str): The unit of temperature, can be either 'celsius' or 'fahrenheit'. Default is 'celsius'.

    Returns:
    str: A string that describes the current weather.
    """

    # This is a mock function, so let's return a mock weather report.
    weather_report = f"The current weather in {location} is 20 degrees {unit}."
    return weather_report


def get_function(
    message: dict[str, str], functions: dict[str, Callable]
) -> dict[str, Any]:
    # Note: the JSON response may not always be valid; be sure to handle errors
    name = message["function_call"]["name"]
    callback = functions[name]
    arguments = json.loads(message["function_call"]["arguments"])
    content = callback(**arguments)
    response = {
        "role": "function",
        "name": name,
        "content": content,
    }
    return response


function_schemas = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

# NOTE: Mixtral is a Instruct based MoE Model which is composed of 8 7B Mistral models
system_message = ChatMessage(
    role="system",
    content="<<SYS>>My name is Mistral. I am a kind and helpful assistant.<</SYS>>\n",
)
user_message = ChatMessage(
    role="user",
    content="[INST] Hello! My name is Austin. What is your name? [/INST]\n",
)
model_message = ChatMessage(
    role="user",
    content="Hello Austin, nice to meet you! I'm Mistral, a helpful assistant. How can I assist you today?",
)
user_message = ChatMessage(
    role="user",
    content="[INST] Can you show me a function definition for the fibonacci sequence? [/INST]\n",
)
messages = [system_message, user_message, model_message, user_message]

# Example usage:
endpoint = "/completion"


def main():
    pretty.install()
    llama_cpp_request = LlamaCppRequest("http://127.0.0.1", "8080")
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    # Create the prompt using the get_prompt_sequence function
    prompt = get_prompt_sequence(messages)
    # Structure the prompt for to stream the request
    data = {"prompt": prompt, "stream": True}
    # Generate the models response
    generator = llama_cpp_request.stream(endpoint, data=data)
    # Handle the models generated response
    response = ""
    for chunk in generator:
        response += chunk["content"]  # append each chunk to the response
        print(chunk["content"], end="")  # print the chunk out to the user
        sys.stdout.flush()  # flush the output to standard output
    print()  # add padding to models output
    # Build the models response
    model_message = ChatMessage(role="user", content=response)
    messages.append(model_message)


if __name__ == "__main__":
    main()
