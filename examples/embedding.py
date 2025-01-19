"""
Embedding Generation Example
==========================

This script demonstrates the generation of embeddings using llama.cpp's REST API. It is not compatible with the OpenAI API endpoint. For OpenAI compatible API endpoint, please refer to `examples/oai-embedding.py`.

Overview
--------

This script is designed to experiment with cosine similarity, normalization, and computing the probability of similarity. The original input is a matrix of n x V, where V is the vocabulary size.

Usage
-----

To use this script, replace the placeholder values with your own data.
"""

from pprint import pprint
from typing import Generator

import numpy as np
import requests


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize(x: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(x)
    return x / magnitude  # normalized vector


def softmax(scores: np.ndarray) -> np.ndarray:
    """Returns the softmax of a vector."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def get_embeddings() -> Generator:
    # Initialize parameters
    url = "http://localhost:8080/embedding"  # Allows pooling set to None
    headers = {"Content-Type": "application/json"}
    payload = {"input": ["Hello, world!", "Another example text"], "model": "my_model"}

    # Send the request
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()  # embeddings is a list of dict
    for result in data:
        matrix = result["embedding"]  # dims is input x vocab_size
        yield np.array(matrix, dtype=np.float32)


def main():
    embeddings = get_embeddings()
    print(embeddings)


if __name__ == "__main__":
    main()
