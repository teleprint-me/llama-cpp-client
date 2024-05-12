"""
Module: llama_cpp_client.tokenizer
"""

import argparse
import pathlib
from typing import List, Optional

from llama_cpp_client.request import LlamaCppRequest


class LlamaCppTokenizer:
    def __init__(self, request: Optional[LlamaCppRequest] = None):
        """
        Initializes the LlamaCppTokenizer with a LlamaCppRequest instance.

        :param server_url: The base URL of the server where the tokenize and detokenize endpoints are available.
        """
        self.llama_cpp_request = request or LlamaCppRequest()

    def tokenize(self, content: str, add_special: bool = False) -> List[int]:
        """
        Tokenizes a given text using the server's tokenize endpoint.

        :param content: The text content to tokenize.
        :return: A list of token IDs.
        """
        data = {"content": content, "add_special": add_special}
        llama_cpp_response = self.llama_cpp_request.post("/tokenize", data=data)
        return llama_cpp_response.get("tokens", [])

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a given sequence of token IDs using the server's detokenize endpoint.

        :param tokens: The list of token IDs to detokenize.
        :return: The detokenized text.
        """
        data = {"tokens": tokens}
        llama_cpp_response = self.llama_cpp_request.post("/detokenize", data=data)
        return llama_cpp_response.get("content", "")


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
    # Initialize the LlamaCppTokenizer instance
    tokenizer = LlamaCppTokenizer(request=llama_cpp_request)

    if args.file:
        path = pathlib.Path(args.prompt)
        with open(path, "r") as file:
            content = file.read()
        encodings = tokenizer.tokenize(content, args.special)
    else:
        encodings = tokenizer.tokenize(args.prompt, args.special)

    if args.encoded:
        print(encodings)

    # Detokenize the tokens
    decodings = tokenizer.detokenize(encodings)
    if args.decoded:
        print(decodings)

    if args.length:
        print(len(encodings))


if __name__ == "__main__":
    main()
