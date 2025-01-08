"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.rag
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

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.embedding import LlamaCppEmbedding, LlamaCppReranker
from llama_cpp_client.logger import get_default_logger


class LlamaCppDatabase:
    """
    Database for storing embeddings and facilitating RAG workflows.
    """

    def __init__(self, api: LlamaCppAPI, db_path: str, verbose: bool = False) -> None:
        """Initialize the database."""
        self.api = api
        self.embedding = LlamaCppEmbedding(self.api, verbose=verbose)
        self.reranker = LlamaCppReranker(self.embedding)  # Optional: Add verbosity flag
        self.logger = get_default_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )

        # Setup database parameters
        self.db_path = db_path
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row

        # Create the embeddings table if it doesn't already exist
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                text TEXT,
                embedding TEXT
            )"""
        )
        self.db.commit()

    def insert_embedding_by_file(self, file_path: str) -> None:
        """
        Insert a new embedding into the database by processing the file.

        Args:
            file_path (str): Path to the file to process and embed.
        """
        # Check if the file already exists in the database
        cursor = self.db.execute(
            "SELECT 1 FROM embeddings WHERE file_path = ?", (file_path,)
        )
        if cursor.fetchone():
            self.logger.debug(f"Embedding for {file_path} already exists.")
            return

        # Process the file and generate embeddings
        embedding = self.embedding.process_file_embedding(file_path)
        with open(file_path, "r") as f:
            text = f.read()

        # Serialize the embedding and insert it
        serialized_embedding = json.dumps(embedding.tolist())
        self.db.execute(
            """INSERT INTO embeddings (file_path, text, embedding) VALUES (?, ?, ?)""",
            (file_path, text, serialized_embedding),
        )
        self.db.commit()
        self.logger.debug(f"Inserted embedding for {file_path}.")

    def insert_embeddings_from_directory(self, dir_path: str) -> None:
        """
        Insert embeddings for all files in a directory.

        Args:
            dir_path (str): Path to the directory containing files to process.
        """
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.insert_embedding_by_file(file_path)

    def search_similar_embeddings(self, query: str, top_n: int = 3) -> List[dict]:
        """
        Search for the top-N most similar embeddings in the database to a given query.

        Args:
            query (str): Query text to search against.
            top_n (int): Number of top results to return.

        Returns:
            List[dict]: List of top-N similar embeddings with their scores.
        """
        # Generate query embedding
        query_embedding = self.embedding.process_embedding(query)

        # Retrieve and compare all embeddings
        rows = self.db.execute(
            "SELECT file_path, text, embedding FROM embeddings"
        ).fetchall()
        results = []
        for row in rows:
            stored_embedding = np.array(json.loads(row["embedding"]))
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            results.append(
                {
                    "file_path": row["file_path"],
                    "text": row["text"],
                    "similarity": similarity,
                }
            )

        # Sort results by similarity and return the top-N
        sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)[
            :top_n
        ]
        self.logger.debug(f"Found {len(sorted_results)} similar embeddings.")
        return sorted_results


def main():
    parser = argparse.ArgumentParser(description="LlamaCpp Database and RAG Workflow")
    parser.add_argument(
        "--database",
        type=str,
        default="data/llama.db",
        help="Path to the SQLite database.",
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
