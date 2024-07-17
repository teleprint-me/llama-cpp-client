"""
Module: llama_cpp_client.client
"""

import argparse
import sys

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
        self.api = api if api is not None else LlamaCppAPI()

        if history is not None:
            self.history = history
        else:
            self.history = LlamaCppHistory(
                session_name="client",
                system_message="My name is Llama. I am a helpful assistant.",
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = LlamaCppTokenizer(self.api.request)

        self.console = Console()

    def _render_completions_once_on_start(self) -> None:
        self.history.load()
        element = ""
        for completion in self.history:
            if completion["role"] == "user":
                element = ""
                element += completion["content"]
            if completion["role"] == "assistant":
                element += completion["content"]
            self.console.print(Markdown(f"**{completion['role']}**"), end="")
            self.console.print(element)
            print()

    def _render_chat_completions_once_on_start(self) -> None:
        self.history.load()
        for completion in self.history:
            self.console.print(Markdown(f"**{completion['role']}**"))
            self.console.print(completion["content"])
            print()

    def stream_completion(self) -> str:
        # NOTE: The API only supports individual completions at the moment
        # Currently researching how to implement multi-prompting
        content = self.history[-1]["content"]
        generator = self.api.completion(content)

        print()  # Pad model output
        self.console.print(Markdown("**completion**"))
        self.console.print(content, end="")
        # Handle the model's generated response
        for response in generator:
            if "content" in response:
                token = response["content"]
                content += token
                self.console.print(token, end="")
                sys.stdout.flush()
        print("\n")  # Pad model output

        return content

    def stream_chat_completion(self) -> str:
        content = ""
        generator = self.api.chat_completion(self.history.completions)

        print()  # Pad model output
        self.console.print(Markdown("**chat completion**"))
        for response in generator:
            if "content" in response["choices"][0]["delta"]:
                token = response["choices"][0]["delta"]["content"]
                content += token
                self.console.print(token, end="")
                sys.stdout.flush()
        print("\n")  # Pad model output

        return content

    def prompt_user(self) -> None:
        try:
            self.console.print(Markdown("**user**"))
            prompt = self.history.prompt()
            self.history.append({"role": "user", "content": prompt})

        # NOTE: Ctrl + C (interrupt) to exit
        except KeyboardInterrupt:
            self.history.save()
            exit()

        # Ctrl + D (eof) to pop a message from history and regenerate completion
        except EOFError:
            if self.history.completions:
                completion = self.history.pop()
                print("\nPopped", completion["role"], "element from history.\n")

    def prompt_assistant(self, completions_type: str) -> None:
        try:
            if completions_type == "completions":
                completion = self.stream_completion()
            elif completions_type == "chat_completions":
                completion = self.stream_chat_completion()
            else:
                raise ValueError(f"Unrecognized completions type: {completions_type}")

            self.history.append({"role": "assistant", "content": completion})
            self.history.save()
        except KeyboardInterrupt:
            if self.history.completions:
                completion = self.history.pop()
                print("\nPopped", completion["role"], "element from history.\n")

    def run(self, completions_type: str):
        if completions_type == "completions":
            self._render_completions_once_on_start()
        elif completions_type == "chat_completions":
            self._render_chat_completions_once_on_start()
        else:
            raise ValueError(f"Unrecognized completions type: {completions_type}")

        while True:
            self.prompt_user()
            self.prompt_assistant(completions_type)

    def run_completions(self) -> None:
        """Convenience method for running completions"""
        self.run("completions")

    def run_chat_completions(self) -> None:
        """Convenience method for running chat completions"""
        self.run("chat_completions")


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
        help="The language models system message. Can be a string or plaintext file.",
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
        llama_cpp_history = LlamaCppHistory(args.session_name)
    else:
        llama_cpp_history = LlamaCppHistory(args.session_name, args.system_message)

    llama_cpp_client = LlamaCppClient(
        api=llama_cpp_api,
        history=llama_cpp_history,
        tokenizer=llama_cpp_tokenizer,
    )
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    if args.completions:
        llama_cpp_client.run_completions()
    else:
        llama_cpp_client.run_chat_completions()


if __name__ == "__main__":
    main()
