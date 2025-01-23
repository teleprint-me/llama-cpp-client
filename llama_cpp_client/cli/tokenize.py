"""
Copyright © 2023 Austin Berrio

Script: llama_cpp_client.cli.tokenize

Description: Experimental script for handling REST API tokenizer requests.

Usage Example:
python -m llama_cpp_client.cli.tokenize 'Hello, world!'
Encoding: [9906, 11, 1917, 0]
"""

import argparse
import pathlib

from llama_cpp_client.common.args import (
    add_common_general_args,
    add_common_request_args,
)
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.request import LlamaCppRequest
from llama_cpp_client.llama.tokenizer import LlamaCppTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="A string or file path used as input")
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="Treat the prompt as a plaintext file",
    )
    parser.add_argument(
        "-p",
        "--with-pieces",
        action="store_true",
        help="Include token pieces in output.",
    )
    parser.add_argument(
        "-s",
        "--add-special-tokens",
        action="store_true",
        help="Include special tokens.",
    )
    parser.add_argument(
        "-d",
        "--decoded",
        action="store_true",
        help="Convert encoding ids to text and print to stdout",
    )
    parser.add_argument(
        "-l",
        "--length",
        action="store_true",
        help="Get the length of the input and print to stdout",
    )
    add_common_request_args(parser)  # Host and port
    add_common_general_args(parser)  # Verbose
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize the LlamaCppRequest instance
    llama_request = LlamaCppRequest(
        base_url=args.base_url, port=args.port, verbose=args.verbose
    )
    # Initialize the LlamaCppAPI instance
    llama_api = LlamaCppAPI(llama_request=llama_request, verbose=args.verbose)
    # Initialize the LlamaCppTokenizer instance
    llama_tokenizer = LlamaCppTokenizer(llama_api=llama_api, verbose=args.verbose)

    # Load the tokenizer model
    if args.file:
        path = pathlib.Path(args.prompt)
        with open(path, "r") as file:
            args.prompt = file.read()

    # Encode the prompt using the tokenizer
    encodings = llama_tokenizer.encode(
        args.prompt, args.add_special_tokens, args.with_pieces
    )
    print(f"Encoding: {encodings}")

    if args.length:
        print(f"Encoding length: {len(encodings)}")

    if args.decoded:
        decodings = llama_tokenizer.decode(encodings)
        print(f"Decoding: {decodings}")


if __name__ == "__main__":
    main()
