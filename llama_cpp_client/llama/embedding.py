"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.embedding

Description: Module for handling language model embeddings.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List

import numpy as np

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.chunker import LlamaCppChunker


class LlamaCppEmbedding:
    """Class for processing and generating embeddings for single or batched inputs."""

    def __init__(self, api: LlamaCppAPI = None, verbose: bool = False) -> None:
        """
        Initialize the LlamaCppEmbedding class.

        Args:
            api (LlamaCppAPI): The API instance for interacting with the model.
            verbose (bool): Whether to enable verbose logging.
        """
        self.api = api if api is not None else LlamaCppAPI()
        self.verbose = verbose

    def process_embedding(self, content: str) -> np.ndarray:
        """
        Generate an embedding for a single input string.

        Args:
            content (str): Input text to generate an embedding for.

        Returns:
            np.ndarray: A normalized embedding vector.
        """
        response = self.api.embedding(content)
        embedding = response[0]["embedding"]
        embedding_vectors = np.array(embedding)
        assert (
            embedding_vectors.shape[1] == self.api.get_embed_size()
        ), "Embedding size mismatch."
        return np.mean(embedding_vectors, axis=0)

    def process_file_embedding(
        self, file_path: str, chunk_size: int = 0, batch_size: int = 512
    ) -> np.ndarray:
        """
        Generate an embedding for a file, optionally chunked.

        Args:
            file_path (str): Path to the file to embed.
            chunk_size (int, optional): Chunk size for splitting the file. Defaults to None.

        Returns:
            np.ndarray: A normalized embedding vector for the entire file.
        """
        chunker = LlamaCppChunker(file_path, api=self.api, verbose=self.verbose)
        chunker.normalize_text()  # Apply minimal normalization
        chunks = chunker.chunk_text_with_model(
            chunk_size=chunk_size, batch_size=batch_size
        )
        embeddings = [self.process_embedding(chunk) for chunk in chunks]
        return np.mean(embeddings, axis=0)

    # TODO: Rename to `process_file_embedding_with_metadata` for clarity.
    def process_file_embedding_entries(
        self, file_path: str, chunk_size: int, batch_size: int
    ) -> List[dict]:
        """Returns a list of embeddings for a file with metadata."""
        chunker = LlamaCppChunker(file_path, api=self.api, verbose=self.verbose)
        chunker.normalize_text()
        chunks = chunker.chunk_text_with_model(
            chunk_size=chunk_size, batch_size=batch_size
        )
        results = []
        for chunk_id, chunk in enumerate(chunks):
            embedding = self.process_embedding(chunk)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                    "embedding": embedding,
                }
            )
        return results


class LlamaCppSimilarity:
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Returns the cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Returns the Euclidean distance between two vectors."""
        return np.linalg.norm(a - b)

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Returns the Manhattan distance between two vectors."""
        return np.sum(np.abs(a - b))

    @staticmethod
    def get_metric(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Returns the metric function for a given metric.

        Supported metrics:
            - "cosine": Cosine similarity
            - "euclidean": Euclidean distance
            - "manhattan": Manhattan distance

        Raises:
            ValueError: If the metric is unsupported.
        """
        metrics = {
            "cosine": LlamaCppSimilarity.cosine_similarity,
            "euclidean": LlamaCppSimilarity.euclidean_distance,
            "manhattan": LlamaCppSimilarity.manhattan_distance,
        }
        if metric not in metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        return metrics[metric]

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """Returns the softmax of a vector."""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    @staticmethod
    def sort_mapping(
        mapping: List[Dict[str, Any]], metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """Returns a sorted list of dictionaries by a given score."""
        reverse = metric == "cosine"  # Descending for cosine, ascending for distances
        return sorted(mapping, key=lambda x: x["score"], reverse=reverse)

    @staticmethod
    def top_n_mapping(
        mapping: List[Dict[str, Any]], metric: str = "cosine", n: int = 3
    ) -> List[Dict[str, Any]]:
        """Returns the top N documents for a given query."""
        return LlamaCppSimilarity.sort_mapping(mapping, metric)[:n]

    @staticmethod
    def normalize_mapping(mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalizes a list of dictionaries by their scores."""
        scores = np.array([d["score"] for d in mapping])
        normalized_scores = LlamaCppSimilarity.softmax(scores)
        for i, score in enumerate(normalized_scores):
            mapping[i]["score"] = score
        return mapping

    @staticmethod
    def rerank_mapping(
        mapping: List[Dict[str, Any]],
        query_embedding: np.ndarray,
        metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """Reranks the mapping based on the given scores."""
        metric_func = LlamaCppSimilarity.get_metric(metric)
        for doc in mapping:
            doc["score"] = metric_func(query_embedding, doc["embedding"])
        return mapping


class LlamaCppDatabase:
    """Database for storing embeddings and facilitating RAG workflows."""

    def __init__(
        self, db_path: str, api: LlamaCppAPI = None, verbose: bool = False
    ) -> None:
        self.api = api if api is not None else LlamaCppAPI()
        self.embedding = LlamaCppEmbedding(self.api, verbose=verbose)
        self.similarity = LlamaCppSimilarity()
        self.logger = get_logger(
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
                self.insert_embedding_from_file(file_path, chunk_size, batch_size)
            self.logger.debug("Inserted embeddings from directory: %s" % root)
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
        top_n: int = 3,
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

        top_results = self.similarity.top_n_mapping(results, metric, n=top_n)
        self.logger.debug(f"Found {len(top_results)} relevant chunks in the database.")
        return top_results

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
        sorted_results = self.similarity.sort_mapping(results, metric)

        # Rerank results (modifies in place)
        self.similarity.rerank_mapping(sorted_results, query_embeddings, metric)

        # Retrieve the top-N results
        top_results = self.similarity.top_n_mapping(sorted_results, metric, n=top_n)
        self.logger.debug(f"Top {top_n} results: {top_results}")
        return top_results
