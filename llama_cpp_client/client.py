"""
Module: llama_cpp_client.client
"""

import argparse
import json
import sys
from typing import Any, Callable

from rich import pretty, print
from rich.console import Console
from rich.markdown import Markdown

from llama_cpp_client.api import LlamaCppAPI
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

    def _render_completions_once_on_start(self) -> None:
        self.history.load()
        element = ""
        for completion in self.history.completions:
            if completion["role"] == "prompt":
                element = ""
                element += completion["content"]
            if completion["role"] == "completion":
                element += completion["content"]
            self.console.print(Markdown(f"**{completion['role']}**"))
            self.console.print(element)
            print()

    def _render_chat_completions_once_on_start(self) -> None:
        self.history.load()
        for completion in self.history.completions:
            self.console.print(Markdown(f"**{completion['role']}**"))
            self.console.print(completion["content"])
            print()

    def stream_completion(self, prompt: str) -> str:
        content = ""
        # API only supports individual completions at the moment
        # Currently researching how to implement multi-prompting
        generator = self.api.completion(prompt)
        self.console.print(self.history.completions[-1]["content"], end="")
        # Handle the model's generated response
        for response in generator:
            if "content" in response:
                content += response["content"]
            self.console.print(response["content"], end="")
            sys.stdout.flush()
        print()  # Pad model output

        return content

    def stream_chat_completion(self) -> str:
        content = ""
        generator = self.api.chat_completion(self.history.completions)

        print()  # Pad model output
        self.console.print(Markdown("**assistant**"))
        for response in generator:
            if "content" in response["choices"][0]["delta"]:
                content += response["choices"][0]["delta"]["content"]
            # markdown = Markdown(content + block)
            self.console.print(content, end="")
            sys.stdout.flush()
        print()  # Pad model output

        return content

    def run_completions(self) -> None:
        self._render_completions_once_on_start()

        while True:
            try:
                self.console.print(Markdown("**prompt**"))
                prompt = self.history.prompt()
                self.history.append({"role": "prompt", "content": prompt})
                self.console.print(Markdown("**completion**"))
                completion = self.stream_completion(prompt)
                self.history.append({"role": "completion", "content": completion})
                self.history.save()
            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            except KeyboardInterrupt:
                if self.history.completions:
                    completion = self.history.pop()
                    print("Popped", completion["role"], "element from history.")
            # Adding EOFError prevents an exception and gracefully exits.
            except EOFError:
                self.history.save()
                exit()

    def run_chat_completions(self) -> None:
        """Feed structured input data for the language model to process."""
        self._render_chat_completions_once_on_start()

        while True:
            try:
                self.console.print(Markdown("**user**"))
                content = self.history.prompt()
                self.history.append({"role": "user", "content": content})
                content = self.stream_chat_completion()
                self.history.append({"role": "assistant", "content": content})
                self.history.save()
            # NOTE: Ctrl + c (keyboard) or Ctrl + d (eof) to exit
            except KeyboardInterrupt:
                completion = self.history.pop()
                print("Popped", completion["role"], "element from history.")
            # Adding EOFError prevents an exception and gracefully exits.
            except EOFError:
                self.history.save()
                exit()


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A command-line interface for interacting with the llama language model."
    )
    parser.add_argument(
        "--session-name",
        type=str,
        required=True,
        help="The caches session name",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default="My name is Llama. I am a supportive and helpful assistant.",
        help="The language models system message",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1",
        help="The servers url (default: http://127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8080",
        help="The servers port (default: 8080)",
    )
    parser.add_argument(
        "--completions",
        action="store_true",
        help="Run a completion with the language model (default: False; run chat completions)",
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
        help="List of stop tokens to ignore (default: empty string; use comma delimited list, no spaces).",
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

    if args.completions:
        args.system_message = None
    llama_cpp_history = LlamaCppHistory(args.session_name, args.system_message)

    llama_cpp_client = LlamaCppClient(
        llama_cpp_api,
        llama_cpp_history,
        llama_cpp_tokenizer,
    )
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    if args.completions:
        llama_cpp_client.run_completions()
    else:
        llama_cpp_client.run_chat_completions()


if __name__ == "__main__":
    main()
