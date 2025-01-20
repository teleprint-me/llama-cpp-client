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
        self.context_size = llama_api.get_context_size()

        # Learnable embedding layer over the Llama embeddings
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Add intermediate dense layers
        self.linear1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Final projection layer to map to the desired output embedding size
        self.projection = nn.Linear(hidden_dim, self.embedding_dim)

        # Add dropout and activations
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

            # Create a torch tensor
            embeddings = torch.tensor(llama_embeddings, dtype=torch.float32)

        # Pass through learnable embedding layer
        embeddings = self.embeddings(torch.arange(len(llama_embeddings)))

        # Apply intermediate layers
        hidden = self.dropout(self.activation(self.linear1(embeddings)))
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
