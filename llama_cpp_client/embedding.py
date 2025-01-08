"""
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
from typing import List, Optional, Tuple

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

    def chunk_text_with_model(self, chunk_size: int = 0, overlap: int = 0) -> List[str]:
        """
        Split text into chunks compatible with the model's embedding size.

        Args:
            chunk_size (int): Maximum number of tokens per chunk.
            overlap (int): Number of tokens to overlap between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        if chunk_size <= 0 or chunk_size > self.embed_size:
            chunk_size = self.embed_size
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        tokens = self.api.tokenize(self.file_contents, with_pieces=False)

        chunks = []
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
    """Embedding functions for LlamaCppClient."""

    def __init__(self, file_path: str, api: LlamaCppAPI) -> None:
        """Read a file and return its contents."""
        self.api = api
        self.file_path = file_path
        self.content = self._read_file_contents()


def process_embedding(content: str, api: LlamaCppAPI) -> np.ndarray:
    """Generate embeddings for a file."""
    response = api.embedding(content)
    # Extract and flatten the embeddings vectors from the response
    embedding = response[0]["embedding"]  # This is a matrix (n x embed_size)
    embedding_vectors = [np.array(vector) for vector in embedding]
    embedding_vector_len = len(embedding_vectors[0])
    assert embedding_vector_len == api.get_embed_size(), "Invalid embedding size"
    # Combine all vectors into one array (optional, depending on your use case)
    return np.mean(embedding_vectors, axis=0)


def process_embedding_list(contents: List[str], api: LlamaCppAPI) -> List[np.ndarray]:
    """Generate embeddings for a list of contents."""
    embeddings = []
    for content in contents:
        embedding_vectors = process_embedding(content, api)
        embeddings.append(embedding_vectors)
    return embeddings


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


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the Euclidean distance between two vectors."""
    return np.linalg.norm(v1 - v2)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_top_n_similar(
    query_embedding: np.ndarray,
    content_embedding: List[np.ndarray],
    content: List[str],
    n: int = 3,
) -> List[str]:
    """Finds the top N most similar content to a given query."""
    similarities = [
        cosine_similarity(query_embedding, content_embedding)
        for embedding in content_embedding
    ]
    top_indices = np.argsort(similarities)[-n:][::-1]
    return [(content[i], similarities[i]) for i in top_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the embeddings. Default: False",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Hello!",
        help="Input text to test the embeddings. Default: 'Hello!'",
    )
    parser.add_argument(
        "--filepath",
        help="Input file to parse. Default: None",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine if input is a string or a file
    if args.filepath and os.path.isfile(args.filepath):
        content = process_file(args.filepath)
    else:
        content = ["Hello, World!", "Hi, Universe!", "Goodbye, Galaxy!"]

    # Initialize the API
    api = LlamaCppAPI()

    # Normalize content
    content = normalize_text(content)
    # Chunk content with model
    content = chunk_text_with_model(content, api)

    # Generate embeddings
    query_embedding = process_embedding(args.input, api)
    content_embeddings = process_embedding_list(content, api)

    # Compute cosine similarity
    similarities = [
        cosine_similarity(query_embedding, embedding)
        for embedding in content_embeddings
    ]

    # Find the most similar content
    index = np.argmax(similarities)
    result = content[index], similarities[index]
    print(
        f"Most similar content to '{args.input}': '{result[0]}' (Score: {result[1]:.4f})"
    )

    if args.visualize:
        visualize_embeddings(content_embeddings, content)
