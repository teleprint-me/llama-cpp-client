"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.model.misty

A simple embedding model to intermediate handling semantic document similarities for GGUF models.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_cpp_client.common.args import (
    add_common_api_args,
    add_common_general_args,
    add_common_request_args,
)
from llama_cpp_client.common.json import load_json, save_json
from llama_cpp_client.llama.api import LlamaCppAPI  # Has llama_api.tokenizer()
from llama_cpp_client.llama.chunker import LlamaCppChunker
from llama_cpp_client.llama.request import LlamaCppRequest


class MistyEmbeddingModel(nn.Module):
    def __init__(
        self, llama_api: LlamaCppAPI, hidden_dim: int = 128, dropout_rate: float = 0.1
    ):
        """
        Initializes the embedding model.
        Args:
            llama_api (LlamaCppAPI): API instance for tokenization and embeddings.
            hidden_dim (int): Number of neurons in the hidden layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.llama_api = llama_api
        self.vocab_size = llama_api.get_vocab_size()
        self.embedding_dim = llama_api.get_embed_size()

        # Learnable embedding layer over the Llama embeddings
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Intermediate dense layers
        self.linear1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Final projection layer to map to the desired output embedding size
        self.projection = nn.Linear(hidden_dim, self.embedding_dim)

        # Regularization and activations
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.uniform_(self.linear1.bias, -0.1, 0.1)
        nn.init.uniform_(self.linear2.bias, -0.1, 0.1)
        nn.init.uniform_(self.projection.bias, -0.1, 0.1)

    def forward(self, token_ids: list[id]) -> torch.Tensor:
        """
        Forward pass to generate embeddings.
        Args:
            token_ids (list[int]): Tokenized input represented as a list of token IDs.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        if len(token_ids) >= self.embedding_dim:
            raise ValueError("Token IDs exceed the embedding dimension.")

        # Convert token IDs to tensor
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Learnable token embeddings (if transitioning to standalone mode)
        embeddings = self.embeddings(token_tensor)

        # Intermediate layers
        hidden = self.dropout(self.activation(self.linear1(embeddings)))
        hidden = nn.LayerNorm(hidden.size()[1:])(hidden)  # Optional layer normalization
        hidden = self.dropout(self.activation(self.linear2(hidden)))

        # Final projection and normalization
        output_embeddings = self.projection(hidden)
        return F.normalize(output_embeddings, p=2, dim=1)

    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and document embeddings.
        Args:
            query_embeddings (torch.Tensor): Tensor of shape (num_queries, embedding_dim).
            document_embeddings (torch.Tensor): Tensor of shape (num_documents, embedding_dim).
        Returns:
            torch.Tensor: Pairwise similarity scores of shape (num_queries, num_documents).
        """
        # Ensure tensors are 2D by removing extra dimensions
        if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
            raise ValueError("Both queries and documents must be 2D tensors.")

        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(
            query_embeddings, p=2, dim=1
        )  # (num_queries, embedding_dim)
        document_norm = F.normalize(
            document_embeddings, p=2, dim=1
        )  # (num_documents, embedding_dim)

        # Compute pairwise cosine similarity using matrix multiplication
        return torch.mm(
            query_norm, document_norm.mT
        )  # Use .mT for proper matrix transpose


# Training Loop
def train_model(
    model: nn.Module,
    training_data: list[dict[str, any]],
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    """
    Train the MistyEmbeddingModel using a simple synthetic dataset.

    Args:
        model (MistyEmbeddingModel): The embedding model to train.
        training_data (list[dict[str, any]]): A list of training data entries.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
    """

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Combine related and unrelated documents
        queries = []
        related_documents = []
        unrelated_documents = []
        for entry in training_data:
            queries.append(entry["query"])
            for item in entry["related"]:
                related_documents.append(item)
            for item in entry["unrelated"]:
                unrelated_documents.append(item)

        # Combine all documents into one list
        documents = related_documents + unrelated_documents

        # Create labels: +1 for related, -1 for unrelated
        labels = torch.cat(
            [torch.ones(len(related_documents)), -torch.ones(len(unrelated_documents))]
        )

        # Generate embeddings
        query_embeddings = model.forward(queries)  # (len(queries), embedding_dim)
        document_embeddings = model.forward(
            documents
        )  # (len(documents), embedding_dim)

        # Compute loss directly on embeddings
        repeated_labels = labels.repeat(len(queries))  # Repeat labels for each query
        expanded_query_embeddings = query_embeddings.unsqueeze(1).expand(
            -1, len(documents), -1
        )
        expanded_document_embeddings = document_embeddings.unsqueeze(0).expand(
            len(queries), -1, -1
        )

        loss = loss_fn(
            expanded_query_embeddings.reshape(-1, model.embedding_dim),
            expanded_document_embeddings.reshape(-1, model.embedding_dim),
            repeated_labels,
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Log loss and sample similarity scores
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        with torch.no_grad():
            similarities = model.compute_similarity(
                query_embeddings, document_embeddings
            )
            print(f"Sample Similarity Scores (Epoch {epoch + 1}):")
            print(similarities[:4, :4])  # Print the first few rows and columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a document similarity model.")
    parser.add_argument("-j", "--json", help="Path to the training data file.")
    parser.add_argument("-m", "--model", help="Path to the model file.")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size.",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Physical batch size for processing (Default: 512; Set by llama-server).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Size of each chunk in tokens (Default: 256; Must be less than batch size).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between chunks in tokens (Default: 0).",
    )
    # set host and port
    add_common_request_args(parser)
    # set model hyperparameters
    add_common_api_args(parser)
    # set verbosity
    add_common_general_args(parser)
    return parser.parse_args()


# Example Usage:
if __name__ == "__main__":
    args = parse_args()

    # Load training data from JSON file.
    training_data = load_json(args.json)

    # Initialize core requests
    llama_request = LlamaCppRequest(args.base_url, args.port, verbose=args.verbose)

    # Initialize core REST API
    stop = [token for token in args.stop.split(",") if token]
    llama_api = LlamaCppAPI(
        request=llama_request,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        n_predict=args.n_predict,
        seed=args.seed,
        cache_prompt=args.cache_prompt,
        stop=stop,
        verbose=args.verbose,
    )

    # Initialize chunker
    llama_chunker = LlamaCppChunker(api=llama_api, verbose=args.verbose)

    # Initialize Misty model
    misty = MistyEmbeddingModel(llama_api=llama_api)

    # Train the model
    train_model(
        misty,
        training_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    similarities = []
    for entry in training_data:
        query = entry.get("query")
        query_embedding = misty.forward(query)
        for item in entry.get("related", []):
            document = item.get("document")
            weighted_score = item.get("weighted_score")
            document_embedding = misty.forward(document)
            results = misty.compute_similarity(query_embedding, document_embedding)
            similarities.append(results)

    # Display results
    for i, query in enumerate(similarities):
        print(f"Query {i + 1}: {query}")
        sorted_indices = similarities[i].argsort(descending=True)
        for rank, idx in enumerate(sorted_indices[:3]):  # Top 3 matches
            print(
                f"  Rank {rank + 1}: Document {idx + 1} - Similarity: {similarities[i, idx]:.4f}"
            )
