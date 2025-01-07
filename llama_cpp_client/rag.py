"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.rag
"""

import argparse
import logging
import sqlite3
from typing import Generator, List

from rich import pretty, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown, Panel

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.history import LlamaCppHistory
from llama_cpp_client.request import LlamaCppRequest


def process_file(
    api: LlamaCppAPI, file_path: str
) -> Generator[List[float], None, None]:
    """
    Process a file and generate embeddings for its content.
    Args:
        api (LlamaCppAPI): Initialized API instance.
        file_path (str): Path to the input file.

    Yields:
        List[float]: A single embedding vector.
    """
    # Check server health
    if api.health.get("status") != "ok":
        raise RuntimeError("API is not healthy.")

    # Get model constraints
    embed_size = api.get_embed_size()
    print(f"Embedding size: {embed_size}")

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Tokenize the content
    tokens = api.tokenize(content, add_special=False)
    print(f"Tokenized content: {tokens} (length: {len(tokens)})")

    # Validate token count against embedding size
    if len(tokens) > embed_size:
        raise ValueError(f"Content exceeds embedding size: {embed_size}")

    # Generate embeddings
    response = api.embedding(content)
    for result in response:
        # Each result contains a list of embedding vectors
        for embedding in result["embedding"]:
            yield embedding


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
        "--filepath",
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
    generator = None
    if args.filepath:
        generator = process_file(api, file_path=args.filepath)
    if generator:
        for vector in generator:
            print(vector)
    else:
        print("No generator found")


if __name__ == "__main__":
    main()
