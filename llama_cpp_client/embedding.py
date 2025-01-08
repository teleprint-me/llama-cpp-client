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
from typing import List

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
        if chunk_size <= 0 or chunk_size > max_tokens:
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
        self, file_path: str, chunk_size: int = 0, batch_size: int = 0
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

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Returns the cosine similarity between two vectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Returns the Euclidean distance between two vectors."""
        return np.linalg.norm(v1 - v2)

    def compute_similarity(
        self, v1: np.ndarray, v2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            v1 (np.ndarray): First embedding vector.
            v2 (np.ndarray): Second embedding vector.
            metric (str): Similarity metric ('cosine' or 'euclidean').

        Returns:
            float: Similarity score.
        """
        if metric == "cosine":
            return LlamaCppEmbedding.cosine_similarity(v1, v2)
        elif metric == "euclidean":
            return LlamaCppEmbedding.euclidean_distance(v1, v2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def find_top_n_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        contents: List[str],
        n: int = 3,
    ) -> List[str]:
        """
        Find the top N most similar content to a query embedding.

        Args:
            query_embedding (np.ndarray): Query embedding vector.
            embeddings (List[np.ndarray]): List of embedding vectors.
            contents (List[str]): List of corresponding content strings.
            n (int): Number of top similar results to return.

        Returns:
            List[str]: Top N similar content strings with similarity scores.
        """
        similarities = [
            self.compute_similarity(query_embedding, embedding)
            for embedding in embeddings
        ]
        top_indices = np.argsort(similarities)[-n:][::-1]
        return [(contents[i], similarities[i]) for i in top_indices]


class LlamaCppReranker:
    """Manual reranking for documents based on a query."""

    def __init__(self, embedding_util: LlamaCppEmbedding):
        """
        Initialize the reranker.

        Args:
            embedding_util (LlamaCppEmbedding): Instance of the embedding utility.
        """
        self.embedding_util = embedding_util

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax to an array of scores."""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[dict]:
        """
        Rerank documents based on their relevance to a query.

        Args:
            query (str): Query string.
            documents (List[str]): List of document strings.
            top_n (int): Number of top results to return.

        Returns:
            List[dict]: List of top-N ranked documents with scores.
        """
        # Generate query embedding
        query_embedding = self.embedding_util.process_embedding(query)

        # Generate embeddings for documents
        document_embeddings = [
            self.embedding_util.process_embedding(doc) for doc in documents
        ]

        # Compute similarity scores
        scores = np.array(
            [
                self.embedding_util.compute_similarity(
                    query_embedding, doc_embedding, metric="cosine"
                )
                for doc_embedding in document_embeddings
            ]
        )

        # Normalize scores with softmax
        probabilities = self.softmax(scores)

        # Sort documents by scores in descending order
        ranked_indices = np.argsort(probabilities)[::-1][:top_n]
        ranked_results = [
            {"document": documents[i], "score": probabilities[i]}
            for i in ranked_indices
        ]

        return ranked_results


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
    api = LlamaCppAPI()
    embedding_util = LlamaCppEmbedding(api=api, verbose=args.verbose)

    if args.filepath and os.path.isfile(args.filepath):
        # Process file embeddings
        file_embedding = embedding_util.process_file_embedding(
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
    query_embedding = embedding_util.process_embedding(args.input)
    print(f"Generated embedding for input text: '{args.input}'")

    if file_embedding is not None:
        # Compute similarity if a file was provided
        similarity = embedding_util.compute_similarity(query_embedding, file_embedding)
        print(f"Similarity between input and file: {similarity:.4f}")
    else:
        print("No file embedding available for similarity computation.")

    # If future batch processing or additional content embeddings exist,
    # `find_top_n_similar` could be used here with a collection of embeddings and contents.


if __name__ == "__main__":
    main()
