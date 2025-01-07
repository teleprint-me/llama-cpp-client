"""
Module: llama_cpp_client/embedding.py

This module contains functions for embedding simple inputs.

- Generate embeddings for simple inputs (e.g., "Hello, World!", "Hi, Universe!").
- Compute pairwise similarity between them.
- Visualize small-scale embeddings to build intuition.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from llama_cpp_client.api import LlamaCppAPI


def process_file(file_path: str) -> list[str]:
    """Read a file and return its contents."""
    with open(file_path, "r") as file:
        return file.readlines()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def process_embeddings(contents: list[str], api: LlamaCppAPI) -> list[np.ndarray]:
    """Generate embeddings for a list of contents."""
    embeddings = []
    for content in contents:
        response = api.embedding(content)
        # Extract and flatten the embeddings vectors from the response
        embedding = response[0]["embedding"]  # This is a matrix (n x embed_size)
        embedding_vectors = [np.array(vector) for vector in embedding]
        embedding_vector_len = len(embedding_vectors[0])
        assert embedding_vector_len == api.get_embed_size(), "Invalid embedding size"
        # Combine all vectors into one array (optional, depending on your use case)
        combined_embedding = np.mean(embedding_vectors, axis=0)
        embeddings.append(combined_embedding)
    return embeddings


def visualize_embeddings(embeddings: list[np.ndarray], labels: list[str]):
    """Visualize embeddings in 2D using PCA."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=label)
    plt.legend()
    plt.title("Embedding Visualization")
    plt.show()


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

    content = None

    # Determine if input is a string or a file
    if os.path.isfile(args.filepath):
        content = process_file(args.input)
    else:
        # Set default content if no file is provided
        content = ["Hello, World!", "Hi, Universe!", "Goodbye, Galaxy!"]

    # Initialize the API
    api = LlamaCppAPI()

    # Generate embeddings
    embeddings = process_embeddings(content, api)

    # Compute pairwise similarity
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similarity between '{content[i]}' and '{content[j]}': {sim:.4f}")

    if args.visualize:
        visualize_embeddings(embeddings, content)
