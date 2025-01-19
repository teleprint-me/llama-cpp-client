"""
Embedding Generation Example
==========================

This script demonstrates the generation of embeddings using llama.cpp's REST API. It is not compatible with the OpenAI API endpoint. For OpenAI compatible API endpoint, please refer to `examples/oai-embedding.py`.

Overview
--------

This script is designed to experiment with cosine similarity, normalization, and computing the probability of similarity. The original input is a matrix of n x V, where V is the vocabulary size. The output is a matrix of n x E where E is the embedding size.

Usage
-----

This script provides a simple environment to query a model, retrieve embeddings, and compare them to synthetic documents using cosine similarity.

$ python examples/embedding.py
"""

import logging
import random

import numpy as np
import requests

logging.basicConfig(level=logging.INFO)


def cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> float:
    """Returns the cosine similarity between two vectors."""
    # Allow broadcasting? Setting keepdims to True causes similarity scores to fail on logging.info(). Don't know why right now.
    norm_a = np.linalg.norm(a, keepdims=False) + epsilon
    norm_b = np.linalg.norm(b, keepdims=False) + epsilon
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)


def softmax_with_metadata(scores: list[dict]) -> list[dict]:
    """Applies softmax to scores and retains metadata."""
    raw_scores = np.array([entry["score"] for entry in scores])
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    # Allow broadcasting
    probabilities = exp_scores / exp_scores.sum(keepdims=True)

    # Add probabilities back to the metadata
    for i, entry in enumerate(scores):
        entry["probability"] = probabilities[i]
    return scores


def generate_synthetic_documents() -> list[str]:
    """Generates synthetic documents for testing."""
    related_documents = [
        "A typical example of a greeting in many programming languages is a string that simply states 'Hello, world!'",
        "The quick brown fox, also known as the 'universal printer,' is a well-known pangram used to test keyboards and typing skills.",
        "In Lewis Carroll's classic tale, Alice's fall down the rabbit hole marks the beginning of a fantastical adventure in Wonderland.",
        "This string is used as a placeholder or demonstration input in machine learning models, particularly those involving natural language processing and text embeddings.",
    ]
    unrelated_documents = [
        "Completely unrelated text about an entirely different topic.",
        "This is another random sentence that has no connection to the queries.",
        "A technical explanation of cosine similarity and embeddings.",
        "A list of items: apple, orange, banana, and grape.",
    ]
    docs = related_documents + unrelated_documents
    random.shuffle(docs)
    return docs


def get_embeddings(inputs: list[str]) -> list[np.ndarray]:
    """Fetch embeddings from the REST API for the given inputs."""
    url = "http://localhost:8080/embedding"
    headers = {"Content-Type": "application/json"}
    payload = {"input": inputs, "model": "my_model"}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    embeddings = []
    for result in data:
        matrix = result.get("embedding")
        if matrix is None:
            raise ValueError(f"Malformed response for input: {inputs}")
        embeddings.append(np.array(matrix, dtype=np.float32))
    return embeddings


def main():
    queries = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Alice went down the rabbit hole.",
        "This is a test input for embeddings.",
    ]

    # Generate synthetic documents
    synthetic_docs = generate_synthetic_documents()
    logging.info("Synthetic documents created.")

    # Get embeddings for queries and synthetic documents
    all_inputs = queries + synthetic_docs
    embeddings = get_embeddings(all_inputs)

    # Split embeddings into queries and synthetic sets
    query_embeddings = embeddings[: len(queries)]
    doc_embeddings = embeddings[len(queries) :]

    # Compare each query to all synthetic documents
    for i, query_embed in enumerate(query_embeddings):
        scores = []
        logging.info(
            f"Query Embedding {i + 1}: min={query_embed.min()}, max={query_embed.max()}, mean={query_embed.mean()}"
        )

        for j, doc_embed in enumerate(doc_embeddings):
            logging.info(
                f"Doc Embedding {j + 1}: min={doc_embed.min()}, max={doc_embed.max()}, mean={doc_embed.mean()}"
            )
            similarity = cosine_similarity(query_embed.flatten(), doc_embed.flatten())
            logging.info(
                f"Query {i + 1} -> Document {j + 1}: Similarity = {similarity:.4f}"
            )
            scores.append(
                {
                    "query": queries[i],
                    "document": synthetic_docs[j],
                    "score": similarity,
                }
            )

        # Apply softmax to the scores for the current query
        results = softmax_with_metadata(scores)
        results.sort(
            key=lambda x: x["probability"], reverse=True
        )  # Sort by probability

        # Log the top match for this query
        top_result = results[0]
        logging.info(
            f"Query {i + 1}: {top_result['query']}, "
            f"Top Match Score: {top_result['score']:.4f}, "
            f"Probability: {top_result['probability']:.4f}, "
            f"Document:\n{top_result['document']}"
        )


if __name__ == "__main__":
    main()
