"""
Script: llama_cpp_client.cli.score
Description:
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
    # Allow broadcasting? Setting keepdims to True causes similarity scores to fail on logging.info(). Don't know why right now.
    norm_a = np.linalg.norm(a) + epsilon
    norm_b = np.linalg.norm(b) + epsilon
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Applies softmax to scores."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


# NOTE: Prompts can be batched, but this simplifies the overall mechanics.
def get_embeddings(llama_api: LlamaCppAPI, prompt: str) -> list[list[float]]:
    response = llama_api.embedding(prompt)
    if "error" in response:
        raise ValueError(response["message"])
    matrix = response[0].get("embedding")
    if matrix is None:
        raise ValueError(f"Malformed response for input: {prompt}")
    return np.array(matrix, dtype=np.float32).flatten()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="JSON file containing semantic training data.",
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
        for related, unrelated in zip(entry["related"], entry["unrelated"]):
            # related documents
            rel_doc = related["document"]
            rel_score = related["score"]
            rel_embed = get_embeddings(llama_api, rel_doc)
            rel_act_score = cosine_similarity(query_embed, rel_embed)
            rel_probs = softmax(np.array([rel_score, rel_act_score], dtype=np.float32))
            rel_probs[::-1].sort()
            print(
                "related document:",
                f"{rel_doc},",
                "related score:",
                f"{rel_score},",
                "actual score:",
                f"{rel_act_score:.2f},",
                "probabilities:",
                f"{rel_probs},",
            )
            # unrelated documents
            unrel_doc = unrelated["document"]
            unrel_score = unrelated["score"]
            unrel_embed = get_embeddings(llama_api, unrel_doc)
            unrel_act_score = cosine_similarity(query_embed, unrel_embed)
            unrel_probs = softmax(
                np.array([unrel_score, unrel_act_score], dtype=np.float32)
            )
            unrel_probs[::-1].sort()
            print(
                "Unrelated Document:",
                f"{unrel_doc},",
                "Unrelated score:",
                f"{unrel_score},",
                "actual score:",
                f"{unrel_act_score:.2f},",
                "probabilities:",
                f"{unrel_probs},",
            )
            probs = softmax(
                np.array(
                    [rel_score, rel_act_score, unrel_score, unrel_act_score],
                    dtype=np.float32,
                )
            )
            probs[::-1].sort()
            print(f"Overall Probabilities: {probs}")


if __name__ == "__main__":
    main()
