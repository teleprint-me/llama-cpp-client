"""
Copyright © 2023 Austin Berrio

Script: llama_cpp_client.cli.rag

Description: Experimental script for performing Retrieval Augmented Generation requests.
"""

import argparse
import json
import logging
import os
import sqlite3
from typing import Generator, List

import numpy as np
from rich import pretty, print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown, Panel

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.embedding import (
    LlamaCppDatabase,
    LlamaCppEmbedding,
    LlamaCppSimilarity,
)


# Retrieval Augmented Generation (RAG)
class LlamaCppRAG:
    # TODO: Add RAG implementation
    pass


def main():
    parser = argparse.ArgumentParser(description="LlamaCpp Database and RAG Workflow")
    parser.add_argument(
        "--database",
        type=str,
        default="data/llama.db",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query embedding for a single file. Default: None",
    )
    parser.add_argument(
        "--insert-file",
        type=str,
        help="Insert embedding for a single file.",
    )
    parser.add_argument(
        "--insert-dir",
        type=str,
        help="Insert embeddings for all files in a directory.",
    )
    parser.add_argument("--query", type=str, help="Query for similarity search.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top results to return.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    # Initialize database and API
    api = LlamaCppAPI()
    database = LlamaCppDatabase(api=api, db_path=args.database, verbose=args.verbose)

    if args.insert_file:
        database.insert_embedding_by_file(args.insert_file)
    elif args.insert_dir:
        database.insert_embeddings_from_directory(args.insert_dir)
    elif args.query:
        results = database.search_similar_embeddings(args.query, top_n=args.top_n)
        print(f"Top {args.top_n} results for query '{args.query}':")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['file_path']} (Score: {result['similarity']:.4f})")
    else:
        print("No action specified. Use --insert-file, --insert-dir, or --query.")


if __name__ == "__main__":
    main()
