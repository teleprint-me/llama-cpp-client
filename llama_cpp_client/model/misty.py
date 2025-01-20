"""
Module: llama_cpp_client.model.misty

A simple embedding model to intermediate handling semantic document similarities for GGUF models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Access the llama tokenizer and llama embeddings
from llama_cpp_client.llama.api import LlamaCppAPI


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

    def forward(self, inputs: list[str]) -> torch.Tensor:
        """
        Forward pass to generate embeddings.
        Args:
            inputs (list[str]): List of text inputs to embed.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        with torch.no_grad():
            # Fetch the embeddings from the Llama API
            response = self.llama_api.embedding(inputs)
            llama_embeddings = [
                result.get("embedding") for result in response if "embedding" in result
            ]

            if not llama_embeddings:
                raise ValueError(f"Malformed response for inputs: {inputs}")

            # Create a torch tensor and remove the extra dimension
            torch_embeddings = torch.tensor(
                llama_embeddings, dtype=torch.float32
            ).squeeze(1)

        # Generate learnable embeddings for the batch
        batch_size = torch_embeddings.size(0)
        learned_embeddings = self.embeddings(torch.arange(batch_size, dtype=torch.long))

        # Combine Llama embeddings with learnable embeddings
        combined_embeddings = torch_embeddings + learned_embeddings

        # Apply intermediate layers
        hidden = self.dropout(self.activation(self.linear1(combined_embeddings)))
        hidden = self.dropout(self.activation(self.linear2(hidden)))

        # Final projection
        output_embeddings = self.projection(hidden)

        # Normalize the embeddings
        return F.normalize(output_embeddings, p=2, dim=1)

    def compute_similarity(
        self, queries: torch.Tensor, documents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and document embeddings.
        Args:
            queries (torch.Tensor): Query embeddings of shape (num_queries, embedding_dim).
            documents (torch.Tensor): Document embeddings of shape (num_documents, embedding_dim).
        Returns:
            torch.Tensor: Pairwise similarity scores of shape (num_queries, num_documents).
        """
        # Ensure tensors are 2D by removing extra dimensions
        if queries.ndim != 2 or documents.ndim != 2:
            raise ValueError("Both queries and documents must be 2D tensors.")

        # Normalize embeddings for cosine similarity
        queries = F.normalize(queries, p=2, dim=1)  # (num_queries, embedding_dim)
        documents = F.normalize(documents, p=2, dim=1)  # (num_documents, embedding_dim)

        # Compute pairwise cosine similarity using matrix multiplication
        return torch.mm(queries, documents.mT)  # Use .mT for proper matrix transpose


# Training Loop
def train_model(
    model: nn.Module,
    queries: list[str],
    related_documents: list[str],
    unrelated_documents: list[str],
    epochs: int = 10,
    lr: float = 0.001,
) -> None:
    """
    Train the MistyEmbeddingModel using a simple synthetic dataset.

    Args:
        model (MistyEmbeddingModel): The embedding model to train.
        queries (list[str]): List of queries.
        related_documents (list[str]): List of documents related to the queries.
        unrelated_documents (list[str]): List of unrelated documents.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.

    Returns:
        None
    """
    # Combine related and unrelated documents
    documents = related_documents + unrelated_documents

    # Create labels: +1 for related, -1 for unrelated
    labels = torch.cat(
        [torch.ones(len(related_documents)), -torch.ones(len(unrelated_documents))]
    )

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

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

        # Logging
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# Example Usage:
if __name__ == "__main__":
    # Initialize LlamaCppAPI instance
    llama_api = LlamaCppAPI()

    # Initialize Misty model
    misty = MistyEmbeddingModel(llama_api=llama_api)

    # Example query and documents
    queries = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Alice went down the rabbit hole.",
        "This is a test input for embeddings.",
    ]
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

    # Train the model
    train_model(
        misty,
        queries,
        related_documents,
        unrelated_documents,
        epochs=10,
        lr=0.001,
    )

    documents = related_documents + unrelated_documents

    # Generate embeddings
    query_embedding = misty.forward(queries)
    documents_embedding = misty.forward(documents)

    # Compute similarities
    similarities = misty.compute_similarity(query_embedding, documents_embedding)

    # Display results
    for i, query in enumerate(queries):
        print(f"Query {i + 1}: {query}")
        sorted_indices = similarities[i].argsort(descending=True)
        for rank, idx in enumerate(sorted_indices[:3]):  # Top 3 matches
            print(
                f"  Rank {rank + 1}: Document {idx + 1} - Similarity: {similarities[i, idx]:.4f}"
            )
