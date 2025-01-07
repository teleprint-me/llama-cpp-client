"""
Module: llama_cpp_client/embedding.py

This module contains functions for embedding simple inputs.

- Generate embeddings for simple inputs (e.g., "Hello, World!", "Hi, Universe!").
- Compute pairwise similarity between them.
- Visualize small-scale embeddings to build intuition.
"""

import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from llama_cpp_client.api import LlamaCppAPI


def process_file(file_path: str) -> List[str]:
    """Read a file and return its contents."""
    with open(file_path, "r") as file:
        return file.readlines()


def normalize_text(content: str) -> str:
    """Normalizes a text by removing punctuation and converting to lowercase."""
    # Strip leading and trailing whitespace
    content = content.strip()
    # Convert to lowercase
    content = content.lower()
    # Remove punctuation
    content = re.sub(r"[^\w\s]", "", content)
    return content


def chunk_text_with_model(
    content: str,
    api: LlamaCppAPI,
    chunk_size: int = None,
    overlap: int = 0,
    verbose: bool = False,
) -> List[str]:
    """
    Splits text into chunks that fit within the model's embedding size, with optional overlap.
    Args:
        content (str): The text to chunk.
        api (LlamaCppAPI): The LlamaCpp API instance.
        chunk_size (int, optional): Maximum number of tokens per chunk. Defaults to model's embedding size.
        overlap (int, optional): Number of tokens to overlap between chunks. Defaults to 0.
    Returns:
        List[str]: List of text chunks.
    """
    # Use the model's embedding size if None or outside of range
    max_embed_size = api.get_embed_size()
    in_range = 0 < chunk_size < max_embed_size
    if chunk_size is None or not in_range:
        chunk_size = max_embed_size

    # Tokenize the content
    tokens = api.tokenize(content, with_pieces=True)

    # Create chunks with overlap
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        # Combine the token pieces into a single string
        chunk_text = " ".join([token["piece"] for token in chunk_tokens])
        chunks.append(chunk_text)
        if verbose:
            print(f"Chunk {i}: length: {len(chunk_tokens)}, {chunk_text}")

    # Handle case where no chunks are generated
    if not chunks and content.strip():
        chunks.append(content.stript())
        if verbose:
            print("Content too short for chunking; returning as single chunk.")

    return chunks


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
