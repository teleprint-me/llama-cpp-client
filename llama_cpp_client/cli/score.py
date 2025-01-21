"""
Script: llama_cpp_client.cli.score
Description: Score semantic relationships using embeddings from a transformer model.
"""

import argparse
import json

import numpy as np

from llama_cpp_client.llama.api import LlamaCppAPI


def load_json(file_path: str) -> list[dict[str, any]]:
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Returns the Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> float:
    """Returns the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a) + epsilon
    norm_b = np.linalg.norm(b) + epsilon
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)


def get_embeddings(llama_api: LlamaCppAPI, prompt: str) -> np.ndarray:
    """Fetch embeddings for a given prompt using the LlamaCppAPI."""
    response = llama_api.embedding(prompt)
    if "error" in response:
        raise ValueError(response["message"])
    matrix = response[0].get("embedding")
    if matrix is None:
        raise ValueError(f"Malformed response for input: {prompt}")
    return np.array(matrix, dtype=np.float32).flatten()


def calc_embeddings(
    llama_api: LlamaCppAPI,
    query_embed: np.ndarray,
    entry: dict[str, any],
    key: str,
) -> None:
    for item in entry.get(key, []):
        document = item["document"]
        score = item["score"]
        doc_embed = get_embeddings(llama_api, document)
        actual_score = cosine_similarity(query_embed, doc_embed)
        distance = euclidean_distance(query_embed, doc_embed)
        mean_score = (score + actual_score) / 2
        print(
            f"Document: {document}, Given Score: {score:.2f}, Actual Score: {actual_score:.2f}, Mean Score: {mean_score:.2f}, Distance: {distance:.2f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score semantic relationships using transformer embeddings."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to JSON file containing semantic training data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llama_api = LlamaCppAPI()

    dataset: list[dict[str, any]] = load_json(args.input)

    for entry in dataset:
        query = entry["query"]
        query_embed = get_embeddings(llama_api, query)
        print(f"\nQuery: {query}")

        print("\nRelated Documents:")
        calc_embeddings(llama_api, query_embed, entry, "related")

        print("\nUnrelated Documents:")
        calc_embeddings(llama_api, query_embed, entry, "unrelated")


if __name__ == "__main__":
    main()
