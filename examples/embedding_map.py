import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)


def cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> float:
    """Returns the cosine similarity between two vectors."""
    # Allow broadcasting? Setting keepdims to True causes similarity scores to fail on logging.info(). Don't know why right now.
    norm_a = np.linalg.norm(a, keepdims=False) + epsilon
    norm_b = np.linalg.norm(b, keepdims=False) + epsilon
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return np.dot(a, b) / (norm_a * norm_b)


def generate_synthetic_documents() -> list[str]:
    """Generates simplified synthetic documents for testing."""
    related_documents = [
        "Hello, world! A simple greeting.",
        "A quick brown fox jumps over a lazy dog.",
        "Alice falls into a rabbit hole.",
        "This is a test example.",
    ]
    unrelated_documents = [
        "Unrelated text about a different topic.",
        "Another random sentence unrelated to the queries.",
        "Technical details about embeddings and their use.",
        "A list of fruits: apple, banana, orange.",
    ]
    return related_documents + unrelated_documents


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
        embeddings.append(np.array(matrix, dtype=np.float32).flatten())
    return embeddings


def visualize_pca(embeddings: list[np.ndarray], labels: list[str], title: str):
    """Visualize embeddings in 2D using PCA."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()


def visualize_heatmap(similarities: np.ndarray, queries: list[str], docs: list[str]):
    """Visualize a heatmap of query-document similarities."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarities,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=[f"Doc {i}" for i in range(len(docs))],
        yticklabels=[f"Query {i}" for i in range(len(queries))],
    )
    plt.title("Query-Document Similarity Heatmap")
    plt.xlabel("Documents")
    plt.ylabel("Queries")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heatmap", action="store_true", help="Visualize similarities with a heatmap"
    )
    parser.add_argument(
        "--pca", action="store_true", help="Visualize embeddings with a scatter plot"
    )
    args = parser.parse_args()

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

    # Compute similarities and visualize heatmap
    if args.heatmap:
        similarities = np.zeros((len(query_embeddings), len(doc_embeddings)))
        for i, query in enumerate(query_embeddings):
            for j, doc in enumerate(doc_embeddings):
                similarities[i, j] = cosine_similarity(query.flatten(), doc.flatten())

        visualize_heatmap(similarities, queries, synthetic_docs)

    # Compute embeddings and visualize space
    elif args.pca:
        # Combine embeddings for PCA
        combined_labels = [f"Query {i}" for i in range(len(queries))] + [
            f"Doc {i}" for i in range(len(synthetic_docs))
        ]

        # Visualize PCA of embeddings
        visualize_pca(
            embeddings,
            combined_labels,
            "Query and Document Embeddings in PCA Space",
        )

    # Log the results to stdout
    else:
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
                similarity = cosine_similarity(
                    query_embed.flatten(), doc_embed.flatten()
                )
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
            # Log the top match for this query
            scores.sort(key=lambda x: x["score"], reverse=True)  # Sort by score

            top_result = scores[0]
            logging.info(
                f"Query {i + 1}: {top_result['query']}, "
                f"Top Match Score: {top_result['score']:.4f}, "
                f"Document:\n{top_result['document']}"
            )


if __name__ == "__main__":
    main()
