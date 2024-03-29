"""
Module: llama_cpp_client.client
"""

import json
import sys
from typing import Any, Callable, Dict, List

from rich import pretty, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from llama_cpp_client.format import get_prompt_sequence
from llama_cpp_client.grammar import LlamaCppGrammar
from llama_cpp_client.history import LlamaCppHistory
from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.tokenizer import LlamaCppTokenizer
from llama_cpp_client.types import ChatMessage


class LlamaCppClient:
    def __init__(
        self,
        request: LlamaCppRequest = None,
        history: LlamaCppHistory = None,
        tokenizer: LlamaCppTokenizer = None,
        top_k: int = 50,
        top_p: float = 0.90,
        min_p: float = 0.1,
        temperature: float = 0.7,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        n_predict: int = -1,
        seed: int = 1337,
        stream: bool = True,
        cache_prompt: bool = True,
    ) -> None:
        # manage llama.cpp client instances
        self.request = request
        self.history = history
        self.tokenizer = tokenizer

        # set llama.cpp server payload
        self.data = {
            "messages": self.chat_history.messages,
            # set model hyper parameters
            "temperature": self.temperature,
            "top_k": self.topk,
            "top_p": self.top_p,
            "n_predict": self.n_predict,
            "seed": self.seed,
            "repeat_penalty": self.repeat_penalty,
            "min_p": self.min_p,
            # set llama.cpp server flags
            "stream": self.stream,
            "cache_prompt": self.cache_prompt,
        }

        self.console = Console()

        self._render_messages_once_on_start()

    def _render_messages_once_on_start(self) -> None:
        self.chat_history.load()
        for message in self.chat_history.messages:
            self.console.print(
                Markdown(message["content"]),
            )

    def encode(self, prompt: List[Dict[str, str]]) -> List[int]: ...

    def decode(self, tokens: List[int]) -> List[Dict[str, str]]: ...

    def health(self):
        try:
            return self.request.get("/health")
        except Exception as e:
            print(f"HealthError: {e}")

    def slots(self):
        try:
            return self.request.get("/slots")
        except Exception as e:
            print(f"SlotsError: {e}")

    def stream_completion(self, prompt):
        block = "█ "
        self.data.prompt = prompt
        # Generate the models response
        generator = llama_cpp_request.stream("/v1/completions", data=self.data)
        with Live(console=self.console) as live:
            content = ""  # concat model response chunks
            for response in generator:
                if "content" in response["choices"][0]["delta"]:
                    content += response["choices"][0]["delta"]["content"]
                if response["choices"][0]["finish_reason"] is not None:
                    block = ""  # Clear the block
                markdown = Markdown(content + block)
                live.update(
                    markdown,
                    refresh=True,
                )
            print()  # add padding to models output

        self.chat_history.append({"role": "assistant", "content": content})

    def prompt_model(self):
        """Feed structured input data for the language model to process."""
        status = self.health()["status"]
        assert status == "ok", "Server not ready or error!"
        self.model_name = self.get_model_name()
        while True:
            try:
                prompt = self.chat_history.prompt()
                self.console.print(Markdown(prompt))
                self.stream_completion(prompt=prompt)

            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            # Adding EOFError prevents an exception and gracefully exits.
            except (KeyboardInterrupt, EOFError):
                self.chat_history.save()
                exit()


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


def main():
    pretty.install()

    # NOTE: Mixtral is a Instruct based MoE Model which is composed of 8 7B Mistral models
    system_message = ChatMessage(
        role="system",
        content="My name is StableLM. I am a supportive and helpful assistant.\n",
    )
    user_message = ChatMessage(
        role="user",
        content="Hello! My name is Austin. What is your name?\n",
    )
    model_message = ChatMessage(
        role="assistant",
        content="Hello Austin, nice to meet you! I'm StableLM, a helpful assistant. How can I assist you today?",
    )
    user_message = ChatMessage(
        role="user",
        content="How can I get of a list of prime number in Python?\n",
    )
    messages = [system_message, user_message, model_message, user_message]

    # Example usage:
    # endpoint = "/completion"
    base_url = "http://127.0.0.1"
    port = "8080"
    endpoint = "/v1/chat/completions"

    llama_cpp_request = LlamaCppRequest(base_url=base_url, port=port)
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    # Structure the prompt for to stream the request
    data = {"messages": messages, "stream": True}
    # Generate the models response
    generator = llama_cpp_request.stream(endpoint, data=data)
    # Handle the models generated response
    content = ""
    for response in generator:
        if "content" in response["choices"][0]["delta"]:
            # extract the token from the response
            token = response["choices"][0]["delta"]["content"]
            # append each chunk to the response
            content += token
            print(token, end="")  # print the chunk out to the user
            sys.stdout.flush()  # flush the output to standard output

    print()  # add padding to models output
    # Build the models response
    model_message = ChatMessage(role="user", content=content)
    messages.append(model_message)


if __name__ == "__main__":
    main()
