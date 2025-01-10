"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.embedding

Description: Module for handling language model embeddings.
"""

import json
import logging
import os
import re
import sqlite3
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI


class FileChunker:
    """Class for processing and chunking file contents."""

    def __init__(
        self, file_path: str, api: LlamaCppAPI = None, verbose: bool = False
    ) -> None:
        """
        Initialize the FileChunker with the API instance and file path.

        Args:
            api (LlamaCppAPI): Instance of the LlamaCppAPI.
            file_path (str): Path to the file to be processed.
            verbose (bool): Whether to enable verbose logging.
        """
        self.api = api if api is not None else LlamaCppAPI()
        self.verbose = verbose
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )
        self.file_contents = ""
        self.read_file_contents(file_path)

    @property
    def embed_size(self) -> int:
        """Return the size of the embedding."""
        return self.api.get_embed_size()

    @property
    def special_token_count(self) -> int:
        """Return the number of special tokens."""
        # Use a dummy input to count special tokens
        return len(self.api.tokenize("", add_special=True, with_pieces=False))

    def read_file_contents(self, file_path: str) -> None:
        """Read a file and store its contents."""
        try:
            with open(file_path, "r") as file:
                self.file_contents = file.read()
            if self.verbose:
                self.logger.debug(f"Read file contents from: {file_path}")
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except IOError as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

    def normalize_text(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        preserve_structure: bool = True,
    ) -> None:
        """
        Normalize text by removing punctuation and converting to lowercase.

        Args:
            lowercase (bool): Whether to convert text to lowercase.
            remove_punctuation (bool): Whether to remove punctuation.
            preserve_structure (bool): Whether to preserve basic punctuation for structure.
        """
        if not self.file_contents.strip():
            self.logger.debug("File is empty; skipping normalization.")
            return

        normalized = self.file_contents.strip()
        if lowercase:
            normalized = normalized.lower()
        if remove_punctuation:
            if preserve_structure:
                normalized = re.sub(r"[^\w\s.,?!'\"()]", "", normalized)
            else:
                normalized = re.sub(r"[^\w\s]", "", normalized)
        self.file_contents = normalized
        if self.verbose:
            self.logger.debug("Text normalization completed.")

    def chunk_text_with_model(
        self,
        chunk_size: int = 0,
        overlap: int = 0,
        batch_size: int = 512,
    ) -> List[str]:
        """
        Split text into chunks compatible with the model's embedding size and batch size.

        Args:
            chunk_size (int): Maximum number of tokens per chunk (defaults to batch size - special tokens).
            overlap (int): Number of tokens to overlap between chunks.
            batch_size (int): Physical batch size (defaults to 512).

        Returns:
            List[str]: List of text chunks.
        """
        # Determine the special token overhead
        max_tokens = batch_size - self.special_token_count
        if not (0 < chunk_size < max_tokens):
            self.logger.debug(
                f"Chunk size adjusted to {max_tokens} to fit within batch constraints."
            )
            chunk_size = max_tokens
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        chunks = []
        tokens = self.api.tokenize(
            self.file_contents, add_special=False, with_pieces=False
        )
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.api.detokenize(chunk_tokens)
            chunks.append(chunk_text)
            if self.verbose:
                self.logger.debug(
                    f"Chunk {i // (chunk_size - overlap)}: {len(chunk_tokens)} tokens."
                )

        if not chunks and self.file_contents.strip():
            chunks.append(self.file_contents.strip())
            self.logger.debug(
                "Content too short for chunking; returned as single chunk."
            )

        self.logger.debug(f"Generated {len(chunks)} chunks.")
        return chunks


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
        chunker = FileChunker(file_path, api=self.api, verbose=self.verbose)
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
        chunker = FileChunker(file_path, api=self.api, verbose=self.verbose)
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


def visualize_embeddings(embeddings: List[np.ndarray], labels: List[str]):
    """Visualize embeddings in 2D using PCA."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=label)
    plt.legend()
    plt.title("Embedding Visualization")
    plt.show()
