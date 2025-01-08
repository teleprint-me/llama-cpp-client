"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client/embedding.py

This module contains functions for embedding simple inputs.

- Generate embeddings for simple inputs (e.g., "Hello, World!", "Hi, Universe!").
- Compute pairwise similarity between them.
- Visualize small-scale embeddings to build intuition.
"""

import argparse
import logging
import os
import re
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from llama_cpp_client.api import LlamaCppAPI
from llama_cpp_client.logger import get_default_logger


class FileChunker:
    """Class for processing and chunking file contents."""

    def __init__(self, api: LlamaCppAPI, file_path: str, verbose: bool = False) -> None:
        """
        Initialize the FileChunker with the API instance and file path.

        Args:
            api (LlamaCppAPI): Instance of the LlamaCppAPI.
            file_path (str): Path to the file to be processed.
            verbose (bool): Whether to enable verbose logging.
        """
        self.api = api
        self.verbose = verbose
        self.logger = get_default_logger(
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

    def __init__(self, api: LlamaCppAPI, verbose: bool = False) -> None:
        """
        Initialize the LlamaCppEmbedding class.

        Args:
            api (LlamaCppAPI): The API instance for interacting with the model.
            verbose (bool): Whether to enable verbose logging.
        """
        self.api = api
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
        chunker = FileChunker(api=self.api, file_path=file_path, verbose=self.verbose)
        chunker.normalize_text()  # Apply minimal normalization
        chunks = chunker.chunk_text_with_model(
            chunk_size=chunk_size, batch_size=batch_size
        )
        embeddings = [self.process_embedding(chunk) for chunk in chunks]
        return np.mean(embeddings, axis=0)

    def process_file_embedding_entries(
        self, file_path: str, chunk_size: int, batch_size: int
    ) -> List[dict]:
        """Returns a list of embeddings for a file with metadata."""
        chunker = FileChunker(api=self.api, file_path=file_path, verbose=self.verbose)
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
    def sort_mapping(mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns a sorted list of dictionaries by a given score."""
        return sorted(mapping, key=lambda x: x["score"], reverse=True)

    @staticmethod
    def top_n_mapping(
        mapping: List[Dict[str, Any]], n: int = 3
    ) -> List[Dict[str, Any]]:
        """Returns the top N documents for a given query."""
        return LlamaCppSimilarity.sort_mapping(mapping)[:n]

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


def main():
    parser = argparse.ArgumentParser(
        description="Embedding generator and similarity search."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Hello, World!",
        help="Input text for generating embeddings. Default: 'Hello, World!'",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Path to a file for embedding generation. Default: None",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk size for splitting the file. Defaults to the model's embedding size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help=(
            "Maximum number of tokens to process per batch. This must match the server's "
            "--ubatch-size setting. Defaults to 512. Adjust as needed based on your "
            "server configuration to avoid input size errors."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top similar content results to display. Default: 3",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    # Initialize the API and embedding utility
    llama_api = LlamaCppAPI()
    llama_embedding = LlamaCppEmbedding(api=llama_api, verbose=args.verbose)
    llama_similarity = LlamaCppSimilarity()

    if args.filepath and os.path.isfile(args.filepath):
        # Process file embeddings
        file_embedding = llama_embedding.process_file_embedding(
            args.filepath,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        print(f"Generated embedding for file: {args.filepath}")
    else:
        # Process single text embedding
        file_embedding = None
        print(f"No valid file provided. Continuing with input text: '{args.input}'")

    # Generate query embedding
    query_embedding = llama_embedding.process_embedding(args.input)
    print(f"Generated embedding for input text: '{args.input}'")

    if file_embedding is not None:
        # Compute similarity if a file was provided
        similarity = llama_similarity.compute_similarity(
            query_embedding, file_embedding
        )
        print(f"Similarity between input and file: {similarity:.4f}")
    else:
        print("No file embedding available for similarity computation.")

    # If future batch processing or additional content embeddings exist,
    # `find_top_n_similar` could be used here with a collection of embeddings and contents.


if __name__ == "__main__":
    main()
