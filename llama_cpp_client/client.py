"""
Module: llama_cpp_client.client
"""

import argparse
import json
from typing import Any, Callable, Dict, List

from rich import pretty, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.format import get_prompt_sequence
from llama_cpp_client.grammar import LlamaCppGrammar
from llama_cpp_client.history import LlamaCppHistory
from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.tokenizer import LlamaCppTokenizer


class LlamaCppClient:
    def __init__(
        self,
        api: LlamaCppAPI = None,
        history: LlamaCppHistory = None,
        tokenizer: LlamaCppTokenizer = None,
    ) -> None:
        # install automatic pretty printing in python repl
        pretty.install()

        # manage llama.cpp client instances
        self.api = api or LlamaCppAPI()
        self.history = history or LlamaCppHistory(
            session_name="client",
            system_message="My name is Llama. I am a supportive and helpful assistant.",
        )
        self.tokenizer = tokenizer or LlamaCppTokenizer(self.api.request)
        self.console = Console()
        self._render_messages_once_on_start()

    @property
    def cursor(self) -> str:
        return "█ "

    def _render_messages_once_on_start(self) -> None:
        self.history.load()
        for message in self.history.messages:
            self.console.print(Markdown(f"**{message['role']}**"))
            self.console.print(message["content"])
            print()

    def stream_completion(self, prompt: str) -> None:
        content = ""
        block = self.cursor
        generator = self.api.completion(prompt)

        # Handle the model's generated response
        with Live(console=self.console) as live:
            for response in generator:
                if "content" in response:
                    content += response["content"]
                if response["stop"]:
                    block = ""  # Clear the block
                live.update(content + block)
        print()  # Pad model output

    def stream_chat_completion(self) -> None:
        content = ""
        block = self.cursor
        generator = self.api.chat_completion(self.history.messages)

        print()  # Pad model output
        self.console.print(Markdown("**assistant**"))
        with Live(console=self.console) as live:
            for response in generator:
                if "content" in response["choices"][0]["delta"]:
                    content += response["choices"][0]["delta"]["content"]
                if response["choices"][0]["finish_reason"] is not None:
                    block = ""  # Clear the block
                # markdown = Markdown(content + block)
                live.update(content + block, refresh=True)
        print()  # Pad model output

        self.history.append({"role": "assistant", "content": content})

    def run_chat(self) -> None:
        """Feed structured input data for the language model to process."""
        while True:
            try:
                self.console.print(Markdown("**user**"))
                content = self.history.prompt()
                self.history.append({"role": "user", "content": content})
                self.stream_chat_completion()
                self.history.save()
            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            except KeyboardInterrupt:
                message = self.history.pop()
                print("Popped", message["role"], "message.")
            # Adding EOFError prevents an exception and gracefully exits.
            except EOFError:
                self.history.save()
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


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A command-line interface for interacting with the llama language model."
    )
    parser.add_argument(
        "-n",
        "--session-name",
        type=str,
        required=True,
        help="The caches session name",
    )
    parser.add_argument(
        "-m",
        "--system-message",
        type=str,
        default="My name is Llama. I am a supportive and helpful assistant.",
        help="The language models system message",
    )
    parser.add_argument(
        "-u",
        "--base-url",
        type=str,
        default="http://127.0.0.1",
        help="The servers url (default: http://127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="8080",
        help="The servers port (default: 8080)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Limit output tokens to top-k most likely (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Only consider tokens with prob greater than top-p (default: 0.9)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="Minimum token probability (default: 0.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for output randomness (default: 0.7)",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.0,
        help="Penalty for repeating tokens (default: 1.0, no effect)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=-1,
        help="The number of tokens to predict (default: -1, inf)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Initial value for random number generator (default: -1, random seed; set to a specific value for reproducible output)",
    )
    parser.add_argument(
        "--cache-prompt",
        action="store_false",
        help="Reuse cached prompts to speed up processing (default: true; set to false to disable)",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="",
        help="List of stop tokens to ignore (default: empty string; use comma delimited list without spaces).",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    llama_cpp_request = LlamaCppRequest(args.base_url, args.port)

    # set models hyperparameters
    stop = [token for token in args.stop.split(",") if token]
    llama_cpp_api = LlamaCppAPI(
        llama_cpp_request,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        n_predict=args.n_predict,
        seed=args.seed,
        cache_prompt=args.cache_prompt,
        stop=stop,
    )
    llama_cpp_tokenizer = LlamaCppTokenizer(llama_cpp_request)
    llama_cpp_history = LlamaCppHistory(args.session_name, args.system_message)
    llama_cpp_client = LlamaCppClient(
        llama_cpp_api, llama_cpp_history, llama_cpp_tokenizer
    )
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    llama_cpp_client.run_chat()


if __name__ == "__main__":
    main()
