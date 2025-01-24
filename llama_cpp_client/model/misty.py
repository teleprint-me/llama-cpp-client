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
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.request import LlamaCppRequest
from llama_cpp_client.llama.tokenizer import LlamaCppTokenizer
from llama_cpp_client.model.dataset import EmbeddingDataset


class EmbeddingModel(nn.Module):
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

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass to generate embeddings.

        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).
            padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len),
                where 1 indicates valid tokens and 0 indicates padding tokens. Default is None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        # Embedding lookup
        embeddings = self.embeddings(x)  # Shape: (batch_size, seq_len, embedding_dim)

        if padding_mask is not None:
            # Apply padding mask by zeroing out embeddings for padding tokens
            embeddings = embeddings * padding_mask.unsqueeze(-1)

        # Pass through the first dense layer
        hidden = self.dropout(self.activation(self.linear1(embeddings)))
        # Optional layer normalization for stable training
        hidden = nn.LayerNorm(hidden.size()[1:])(hidden)
        # Pass through the second dense layer
        hidden = self.dropout(self.activation(self.linear2(hidden)))

        # Final projection to output embedding space
        output_embeddings = self.projection(hidden)
        # Aggregate along sequence dimension (e.g., mean pooling)
        output_embeddings = output_embeddings.mean(dim=1)

        # Normalize output embeddings to unit vectors
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
        if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
            raise ValueError("Both queries and documents must be 2D tensors.")

        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        document_norm = F.normalize(document_embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity using matrix multiplication
        return torch.mm(query_norm, document_norm.mT)


def train_model(
    model_path: str,
    embedding_model: nn.Module,
    batched_dataset: list[dict[str, torch.Tensor]],
    save_every: int = 10,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    """
    Train the EmbeddingModel using a simple synthetic dataset.

    Args:
        model_path (str): Path to save the trained model.
        embedding_model (nn.Module): The embedding model to train.
        batched_dataset (list[dict[str, torch.Tensor]]): List of batched data.
        save_every (int): Interval for saving the model.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=learning_rate)
    loss_fn = nn.CosineEmbeddingLoss()

    embedding_model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(batched_dataset):
            optimizer.zero_grad()

            # Extract tokens and labels
            tokens = batch["tokens"]  # Shape: (batch_size, seq_len)
            labels = batch["labels"]  # Shape: (batch_size, )

            # Generate embeddings
            embeddings = embedding_model(tokens)

            # Create synthetic targets for cosine embedding loss
            # Here, positive samples (1) are assumed for simplicity
            targets = torch.ones(labels.size(0), device=labels.device)

            # Compute loss
            loss = loss_fn(embeddings, embeddings, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

        # Save the model periodically
        if (epoch + 1) % save_every == 0:
            torch.save(embedding_model.state_dict(), model_path)
            print(f"Model saved to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a document similarity model.")
    parser.add_argument(
        "-d", "--dataset-path", help="Path to the training dataset file."
    )
    parser.add_argument("-m", "--model-path", help="Path to the model file.")
    parser.add_argument(
        "-s", "--save-every", type=int, default=10, help="Save model every x epochs."
    )
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

    llama_tokenizer = LlamaCppTokenizer(llama_api, args.verbose)
    embedding_dataset = EmbeddingDataset(llama_tokenizer, args.verbose)
    dataset = embedding_dataset.load(args.dataset_path)
    pad_token_id = llama_tokenizer.encode(args.pad_token, add_special_tokens=False)[0]
    dataset = embedding_dataset.tokenize(
        dataset, max_length=args.max_length, pad_token_id=pad_token_id
    )
    dataset = embedding_dataset.batch(dataset, batch_size=args.batch_size)

    # Initialize Misty model
    embedding_model = EmbeddingModel(llama_api=llama_api)

    # Train the model
    train_model(
        args.model_path,
        embedding_model,
        dataset,
        save_every=args.save_every,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    similarities = []
    for entry in dataset:
        query = entry.get("query")
        query_embedding = embedding_model.forward(query)
        for item in entry.get("related", []):
            document = item.get("document")
            document_embedding = embedding_model.forward(document)
            results = embedding_model.compute_similarity(
                query_embedding, document_embedding
            )
            similarities.append(results)

    # Display results
    for i, query in enumerate(similarities):
        print(f"Query {i + 1}: {query}")
        sorted_indices = similarities[i].argsort(descending=True)
        for rank, idx in enumerate(sorted_indices[:3]):  # Top 3 matches
            print(
                f"  Rank {rank + 1}: Document {idx + 1} - Similarity: {similarities[i, idx]:.4f}"
            )
