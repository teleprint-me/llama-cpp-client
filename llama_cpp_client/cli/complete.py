"""
Copyright © 2023 Austin Berrio

Script: llama_cpp_client.cli.complete

Description: CLI script to perform completions using the llama.cpp server and REST API.
"""

import argparse

import rich.box

from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.client import LlamaCppClient
from llama_cpp_client.llama.history import LlamaCppHistory
from llama_cpp_client.llama.request import LlamaCppRequest

# Create a mapping between the box type enumerations and human readable strings
BOX_TO_STRING = {
    "ascii": rich.box.ASCII,
    "markdown": rich.box.MARKDOWN,
    "minimal": rich.box.MINIMAL,
    "rounded": rich.box.ROUNDED,
    "simple": rich.box.SIMPLE,
    "square": rich.box.SQUARE,
}
BOX_CHOICES = tuple(BOX_TO_STRING.keys())
DEFAULT_BOX = "minimal"


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
    parser.add_argument(
        "--box",
        type=str,
        default=DEFAULT_BOX,
        choices=BOX_CHOICES,
        help="The box type to use for displaying the output. Default: 'minimal'.",
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

    if args.completions:
        llama_cpp_history = LlamaCppHistory(args.session_name)
    else:
        llama_cpp_history = LlamaCppHistory(args.session_name, args.system_message)

    box_type = BOX_TO_STRING[args.box]

    llama_cpp_client = LlamaCppClient(
        api=llama_cpp_api,
        history=llama_cpp_history,
        box=box_type,
    )
    # `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    if args.completions:
        llama_cpp_client.run_completions()
    else:
        llama_cpp_client.run_chat_completions()


if __name__ == "__main__":
    main()
