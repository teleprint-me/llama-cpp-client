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
from llama_cpp_client.embedding import LlamaCppEmbedding, LlamaCppSimilarity
from llama_cpp_client.logger import get_default_logger


class LlamaCppDatabase:
    """Database for storing embeddings and facilitating RAG workflows."""

    def __init__(
        self, db_path: str, api: LlamaCppAPI = None, verbose: bool = False
    ) -> None:
        self.api = api if api is not None else LlamaCppAPI()
        self.embedding = LlamaCppEmbedding(self.api, verbose=verbose)
        self.similarity = LlamaCppSimilarity()
        self.logger = get_default_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )
        self.db_path = db_path
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                chunk TEXT NOT NULL,
                embedding TEXT NOT NULL,
                UNIQUE(file_path, chunk_id)
            )"""
        )
        self.db.commit()
        self.logger.debug("Database created at: %s" % self.db_path)

    def insert_embedding_from_file(
        self, file_path: str, chunk_size: int = 0, batch_size: int = 512
    ) -> None:
        """Insert a new embedding into the database by processing the file."""
        # Check if the file already exists in the database
        cursor = self.db.execute(
            "SELECT 1 FROM embeddings WHERE file_path = ?", (file_path,)
        )
        if cursor.fetchone():
            self.logger.debug(f"Embedding for {file_path} already exists.")
            return

        # Process the file and generate embeddings
        chunked_embeddings = self.embedding.process_file_embedding_entries(
            file_path, chunk_size, batch_size
        )

        # Serialize the embedding and insert it
        for embedding in chunked_embeddings:
            chunk_id = embedding["chunk_id"]
            chunk = embedding["chunk"]
            embedding = embedding["embedding"]
            serialized_embedding = json.dumps(embedding.tolist())
            self.db.execute(
                """
                INSERT INTO embeddings (file_path, chunk_id, chunk, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (file_path, chunk_id, chunk, serialized_embedding),
            )
        self.db.commit()
        self.logger.debug(f"Inserted embedding for {file_path}.")

    def insert_embeddings_from_directory(
        self, dir_path: str, chunk_size: int = 0, batch_size: int = 512
    ) -> None:
        """Insert embeddings for all files in a directory."""
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.insert_embedding_by_file(file_path, chunk_size, batch_size)
        self.logger.debug("Inserted embeddings from directory: %s" % dir_path)

    def query_embeddings(self, query: str) -> np.ndarray:
        """Generate embeddings for the given query."""
        query_embeddings = self.embedding.process_embedding(query)
        self.logger.debug(
            f"Generated query embedding with size: {query_embeddings.shape}"
        )
        return query_embeddings

    def search_embeddings(
        self,
        query_embeddings: np.ndarray,
        metric: str = "cosine",
        normalize_scores: bool = False,
    ) -> List[dict]:
        """Search for embeddings in the database that match a given query."""
        rows = self.db.execute(
            "SELECT file_path, chunk_id, chunk, embedding FROM embeddings"
        ).fetchall()

        # Compute similarity scores
        results = []
        metric_func = self.similarity.get_metric(metric=metric)
        for row in rows:
            stored_embedding = np.array(json.loads(row["embedding"]))
            score = metric_func(query_embeddings, stored_embedding)
            self.logger.debug(
                f"Computed score: {score} for chunk ID: {row['chunk_id']}"
            )
            results.append(
                {
                    "file_path": row["file_path"],
                    "chunk_id": row["chunk_id"],
                    "chunk": row["chunk"],
                    "score": score,
                }
            )

        if normalize_scores:
            self.similarity.normalize_mapping(results)

        self.logger.debug(f"Found {len(results)} relevant chunks in the database.")
        return results

    def rerank_embeddings(
        self,
        query_embeddings: np.ndarray,
        metric: str = "cosine",
        normalize_scores: bool = False,
        top_n: int = 3,
    ) -> List[dict]:
        """Search for the top-N most similar embeddings in the database to a given query."""
        # Perform the search
        results = self.search_embeddings(query_embeddings, metric, normalize_scores)
        sorted_results = self.similarity.sort_mapping(results)

        # Rerank results (modifies in place)
        self.similarity.rerank_mapping(sorted_results, query_embeddings, metric)

        # Retrieve the top-N results
        top_results = self.similarity.top_n_mapping(sorted_results, n=top_n)
        self.logger.debug(f"Top {top_n} results: {top_results}")
        return top_results


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
