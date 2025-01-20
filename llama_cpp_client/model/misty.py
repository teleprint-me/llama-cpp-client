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
    def __init__(self, llama_api: LlamaCppAPI):
        """
        Initializes the embedding model.
        Args:
            llama_api (LlamaCppAPI): API instance for tokenization and embeddings.
            embedding_dim (int): Dimensionality of the final embedding space.
        """

        super().__init__()
        self.llama_api = llama_api
        self.vocab_size = llama_api.get_vocab_size()
        self.embedding_dim = llama_api.get_embed_size()
        self.context_size = llama_api.get_context_size()
        # Projection layer to map the Llama embeddings to the desired embedding dimension
        # This layer is used to reduce the dimensionality of the embeddings to the desired size.
        self.projection = nn.Linear(self., self.embedding_dim)
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        # Initialize the projection layer with Xavier uniform weights and biases
        nn.init.xavier_uniform_(self.projection.weight)
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
            # Fetch the embeddings from the API
            response = self.llama_api.embedding(inputs)
            llama_embeddings = [
                result.get("embedding") for result in response if "embedding" in result
            ]

            if not llama_embeddings:
                raise ValueError(f"Malformed response for inputs: {inputs}")

            # Directly create a torch tensor for efficiency
            embeddings = torch.tensor(llama_embeddings, dtype=torch.float32)

            # Project to the desired dimensionality
            return self.projection(embeddings)

    def backward(self, grad_output: torch.Tensor) -> None:
        """
        Backward pass to compute gradients.
        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.
        """
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            query_embedding = misty.forward(queries)
            document_embedding = misty.forward(documents)
            loss = loss_fn(query_embedding, document_embedding, labels)
            loss.backward()
            optimizer.step()

    def compute_similarity(
        self, query: torch.Tensor, documents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and document embeddings.
        Args:
            query (torch.Tensor): Query embedding of shape (batch_size, embedding_dim).
            documents (torch.Tensor): Document embedding of shape (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Similarity scores of shape (batch_size, 1).
        """
        return F.cosine_similarity(query, documents, eps=1e-8)


# Example Usage:
if __name__ == "__main__":
    # Initialize LlamaCppAPI instance
    llama_api = LlamaCppAPI()

    # Initialize Misty model
    misty = MistyEmbeddingModel(llama_api=llama_api)

    # Example query and documents
    query = ["Find information on quantum physics."]
    documents = [
        "Quantum mechanics explains the behavior of matter and energy.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    # Generate embeddings
    query_embedding = misty.forward(query)
    documents_embedding = misty.forward(documents)

    # Compute similarities
    similarities = misty.compute_similarity(query_embedding, documents_embedding)
    print(similarities)
