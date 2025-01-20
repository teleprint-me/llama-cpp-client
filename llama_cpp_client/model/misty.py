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

    # def backward(self, grad_output: torch.Tensor) -> None:
    #     """
    #     Backward pass to compute gradients.
    #     Args:
    #         grad_output (torch.Tensor): Gradient of the loss with respect to the output.
    #     """
    #     optimizer = torch.optim.Adam(self.projection.parameters(), lr=0.001)

    #     for epoch in range(num_epochs):
    #         optimizer.zero_grad()
    #         query_embedding = misty.forward(queries)
    #         document_embedding = misty.forward(documents)
    #         loss = loss_fn(query_embedding, document_embedding, labels)
    #         loss.backward()
    #         optimizer.step()

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
