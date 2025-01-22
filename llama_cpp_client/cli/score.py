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
    print(f"JSON loaded from {file_path}")
    return data


def save_json(data: object, file_path: str) -> None:
    """Dump data to a JSON file with support for NumPy types."""

    def default_serializer(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)  # Convert NumPy floats to Python floats
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert NumPy integers to Python integers
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)
    print(f"JSON saved to {file_path}")


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


def normalize(value: float, min_value: float, max_value: float) -> float:
    """Normalize a value to the range [0, 1]."""
    return (value - min_value) / (max_value - min_value)


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
    entry: dict[str, any],
    key: str,
    weight_synthetic: float = 0.7,
    weight_actual: float = 0.3,
    penalty_factor: float = 0.1,
    max_distance: float = 1.0,
) -> None:
    results = []
    query = entry["query"]
    query_embed = get_embeddings(llama_api, query)
    for item in entry.get(key, []):
        document = item["document"]
        synthetic_score = item["score"]
        doc_embed = get_embeddings(llama_api, document)

        # Calculate metrics
        actual_score = cosine_similarity(query_embed, doc_embed)
        distance = euclidean_distance(query_embed, doc_embed)

        # Normalize distance
        normalized_distance = normalize(distance, 0, max_distance)

        # Weighted score with dynamic adjustment
        dynamic_weight_synthetic = 1 - normalized_distance
        dynamic_weight_actual = normalized_distance
        weighted_score = (
            dynamic_weight_synthetic * synthetic_score
            + dynamic_weight_actual * actual_score
        )

        # Apply penalty for distance
        adjusted_score = weighted_score - penalty_factor * normalized_distance

        # Classify results
        if adjusted_score > 0.8:
            classification = "high"
        elif adjusted_score > 0.5:
            classification = "medium"
        else:
            classification = "low"

        results.append(
            {
                "document": document,
                "synthetic_score": synthetic_score,
                "actual_score": actual_score,
                "weighted_score": weighted_score,
                "adjusted_score": adjusted_score,
                "distance": distance,
                "normalized_distance": normalized_distance,
                "classification": classification,
            }
        )
    entry[key] = results


def print_embeddings(entry: dict[str, any], key: str) -> None:
    print(f"\n{key.capitalize()} Documents:")
    print(f"\nQuery: {entry['query']}")
    for item in entry.get(key, []):
        print(
            f"Document: {item['document']}, "
            f"Synthetic Score: {item['synthetic_score']:.2f}, "
            f"Actual Score: {item['actual_score']:.2f}, "
            f"Weighted Score: {item['weighted_score']:.2f}, "
            f"Adjusted Score: {item['adjusted_score']:.2f}, "
            f"Distance: {item['distance']:.2f}, "
            f"Normalized Distance: {item['normalized_distance']:.2f}, "
            f"Classification: {item['classification']}"
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
        help="Path to JSON file loading synthetic semantic training data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to JSON file saving synthetic semantic training data.",
    )
    parser.add_argument(
        "--weight_synthetic",
        type=float,
        default=0.7,
        help="Weight for synthetic scores.",
    )
    parser.add_argument(
        "--penalty_factor",
        type=float,
        default=0.1,
        help="Penalty factor for distance.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=1.0,
        help="Maximum distance for normalization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llama_api = LlamaCppAPI()

    dataset: list[dict[str, any]] = load_json(args.input)

    for entry in dataset:
        calc_embeddings(
            llama_api,
            entry,
            "related",
            weight_synthetic=args.weight_synthetic,
            weight_actual=1 - args.weight_synthetic,
            penalty_factor=args.penalty_factor,
            max_distance=args.max_distance,
        )
        print_embeddings(entry, "related")

        calc_embeddings(
            llama_api,
            entry,
            "unrelated",
            weight_synthetic=args.weight_synthetic,
            weight_actual=1 - args.weight_synthetic,
            penalty_factor=args.penalty_factor,
            max_distance=args.max_distance,
        )
        print_embeddings(entry, "unrelated")

    if args.output:
        save_json(dataset, args.output)


if __name__ == "__main__":
    main()
