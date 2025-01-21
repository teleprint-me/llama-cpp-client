"""
Script: llama_cpp_client.cli.score
Description:
Improved script for scoring semantic relationships using embeddings from a transformer model.
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


def cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> float:
    """Returns the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a) + epsilon
    norm_b = np.linalg.norm(b) + epsilon
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Applies softmax to scores with optional temperature scaling."""
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    return exp_scores / exp_scores.sum()


# NOTE: Prompts can be batched, but this simplifies the overall mechanics.
def get_embeddings(llama_api: LlamaCppAPI, prompt: str) -> np.ndarray:
    """Fetch embeddings for a given prompt using the LlamaCppAPI."""
    response = llama_api.embedding(prompt)
    if "error" in response:
        raise ValueError(response["message"])
    matrix = response[0].get("embedding")
    if matrix is None:
        raise ValueError(f"Malformed response for input: {prompt}")
    return np.array(matrix, dtype=np.float32).flatten()


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
        print(f"Query: {query}")

        for related in entry.get("related", []):
            rel_doc = related["document"]
            rel_score = related["score"]
            rel_embed = get_embeddings(llama_api, rel_doc)
            rel_act_score = cosine_similarity(query_embed, rel_embed)
            rel_probs = softmax(np.array([rel_score, rel_act_score], dtype=np.float32))
            print(
                "Related Document:",
                f"{rel_doc},",
                "Given Score:",
                f"{rel_score},",
                "Actual Score:",
                f"{rel_act_score:.2f},",
                "Probabilities:",
                f"{rel_probs}",
            )

        for unrelated in entry.get("unrelated", []):
            unrel_doc = unrelated["document"]
            unrel_score = unrelated["score"]
            unrel_embed = get_embeddings(llama_api, unrel_doc)
            unrel_act_score = cosine_similarity(query_embed, unrel_embed)
            unrel_probs = softmax(
                np.array([unrel_score, unrel_act_score], dtype=np.float32)
            )
            print(
                "Unrelated Document:",
                f"{unrel_doc},",
                "Given Score:",
                f"{unrel_score},",
                "Actual Score:",
                f"{unrel_act_score:.2f},",
                "Probabilities:",
                f"{unrel_probs}",
            )

        all_scores = (
            [related["score"] for related in entry.get("related", [])]
            + [
                cosine_similarity(
                    query_embed, get_embeddings(llama_api, related["document"])
                )
                for related in entry.get("related", [])
            ]
            + [unrelated["score"] for unrelated in entry.get("unrelated", [])]
            + [
                cosine_similarity(
                    query_embed, get_embeddings(llama_api, unrelated["document"])
                )
                for unrelated in entry.get("unrelated", [])
            ]
        )

        overall_probs = softmax(np.array(all_scores, dtype=np.float32))
        print(f"Overall Probabilities: {overall_probs}")


if __name__ == "__main__":
    main()
