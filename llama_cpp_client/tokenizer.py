"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.tokenizer
"""

import argparse
import pathlib

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.request import LlamaCppRequest


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="A string or file path used as input")
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
        "-f",
        "--file",
        action="store_true",
        help="Treat the prompt as a plaintext file",
    )
    parser.add_argument(
        "-s",
        "--special",
        action="store_true",
        help="Include special tokens.",
    )
    parser.add_argument(
        "-e",
        "--encoded",
        action="store_true",
        help="Convert text prompt to encoding ids and print to stdout",
    )
    parser.add_argument(
        "-d",
        "--decoded",
        action="store_true",
        help="Convert text prompt to encoding ids and print to stdout",
    )
    parser.add_argument(
        "-l",
        "--length",
        action="store_true",
        help="Get the length of the input and print to stdout",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    # Initialize the LlamaCppRequest instance
    llama_cpp_request = LlamaCppRequest(base_url=args.base_url, port=args.port)
    # Initialize the LlamaCppAPI instance
    llama_cpp_api = LlamaCppAPI(request=llama_cpp_request)

    if args.file:
        path = pathlib.Path(args.prompt)
        with open(path, "r") as file:
            content = file.read()
        encodings = llama_cpp_api.tokenize(content, args.special)
    else:
        encodings = llama_cpp_api.tokenize(args.prompt, args.special)

    if args.encoded:
        print(encodings)

    # Detokenize the tokens
    decodings = llama_cpp_api.detokenize(encodings)
    if args.decoded:
        print(decodings)

    if args.length:
        print(len(encodings))


if __name__ == "__main__":
    main()
