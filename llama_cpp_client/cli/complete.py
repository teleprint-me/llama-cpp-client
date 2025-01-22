"""
Copyright © 2023 Austin Berrio

Script: llama_cpp_client.cli.complete

Description: CLI script to perform completions using the llama.cpp server and REST API.
"""

import argparse

import rich.box

from llama_cpp_client.common.args import (
    add_common_api_args,
    add_common_general_args,
    add_common_request_args,
)
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
        "--box",
        type=str,
        default=DEFAULT_BOX,
        choices=BOX_CHOICES,
        help="The box type to use for displaying the output. Default: 'minimal'.",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="The caches session name",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="My name is Llama. I am a supportive and helpful assistant.",
        help="System message for chat completions. Can be a string or plaintext file.",
    )
    parser.add_argument(
        "--completions",
        action="store_true",
        help="Run a completion with the language model (default: False; run chat completions)",
    )
    add_common_request_args(parser)
    add_common_api_args(parser)
    add_common_general_args(parser)
    return parser.parse_args()


def main():
    args = get_arguments()

    llama_cpp_request = LlamaCppRequest(args.base_url, args.port, verbose=args.verbose)

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
        verbose=args.verbose,
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
