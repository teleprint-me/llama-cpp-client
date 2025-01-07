"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.rag
"""

import argparse
import logging
import sqlite3

from rich import pretty, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown, Panel

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.history import LlamaCppHistory
from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.tokenizer import LlamaCppTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama C++ Client")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose logging. Default: False",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--database",
        help="Database file path. Default: llama.db",
        default="data/llama.db",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Input file to parse. Default: data/plaintext/test.txt",
        default="data/plaintext/test.txt",
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the directory to parse. Default: None",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    api = LlamaCppAPI(log_level=log_level)
    if api.health["status"] != "ok":
        print("LlamaCppAPI is not healthy")
        return


if __name__ == "__main__":
    main()
