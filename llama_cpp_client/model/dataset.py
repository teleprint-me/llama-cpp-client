"""
Module: llama_cpp_client.model.dataset
Description: This module provides functions to load and process datasets for an Embedding model.
"""

import json
import logging
from typing import Dict, List, Union

import torch

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.tokenizer import LlamaCppTokenizer


class EmbeddingDataset:
    def __init__(
        self, llama_tokenizer: LlamaCppTokenizer = None, verbose: bool = False
    ):
        self.tokenizer = (
            llama_tokenizer if llama_tokenizer else LlamaCppTokenizer(verbose=verbose)
        )
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def validate(
        self, dataset: List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]]
    ) -> None:
        """Validate the structure of the dataset."""
        if not isinstance(dataset, list):
            raise ValueError("Dataset should be a list of entries.")
        for entry in dataset:
            if not isinstance(entry, dict):
                raise ValueError("Each entry in the dataset should be a dictionary.")

            for key, value in entry.items():
                if not isinstance(key, str):
                    raise ValueError("Keys in the dataset should be strings.")
                if not isinstance(value, (str, list)):
                    raise ValueError(
                        f"Values in the dataset should be either str or list, but got {type(value)}"
                    )

                if isinstance(value, list):
                    for item in value:
                        if not isinstance(item, dict):
                            raise ValueError(
                                "Each item in the list should be a dictionary."
                            )
                        if "document" not in item or "label" not in item:
                            raise ValueError(
                                "Each item must contain 'document' and 'label' keys."
                            )
                        if not isinstance(item["document"], str):
                            raise ValueError("The 'document' field must be a string.")
                        if not isinstance(item["label"], int):
                            raise ValueError("The 'label' field must be an integer.")

    def load(self, file_path: str) -> List[Dict[str, List[Dict[str, Union[str, int]]]]]:
        with open(file_path, "r") as f:
            try:
                dataset = json.load(f)
                self.validate(dataset)
                return dataset
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error loading dataset: {e}")
                return []

    def save(
        self, file_path: str, dataset: List[Dict[str, Union[torch.Tensor, List]]]
    ) -> None:
        with open(file_path, "w") as f:

            def default(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                return obj

            json.dump(dataset, f, indent=2, default=default)
        self.logger.info(f"Dataset saved to {file_path}")

    def tokenize(
        self,
        dataset: List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]],
        max_length: int = 256,
        pad_token_id: int = -1,
    ) -> List[Dict[str, Union[int, List[int]]]]:
        tokenized_data = []
        for entry in dataset:
            # Read the query
            query = entry["query"]
            query_tokens = self.tokenizer.encode(query, add_special_tokens=False)

            # Process related and unrelated documents
            for relation_type in ["related", "unrelated"]:
                for item in entry[relation_type]:
                    document = item.get("document", "")
                    label = item.get("label")
                    if label is None:
                        raise ValueError("Missing label for document.")

                    # Tokenize document
                    document_tokens = self.tokenizer.encode(
                        document, add_special_tokens=False
                    )

                    # Concatenate query and document tokens
                    tokens = query_tokens + document_tokens
                    # Truncate and pad the sequence
                    tokens = tokens[:max_length] + [pad_token_id] * max(
                        0, max_length - len(tokens)
                    )

                    # Add to tokenized data
                    tokenized_data.append({"tokens": tokens, "label": label})
        return tokenized_data

    def batch(
        self,
        tokenized_data: List[Dict[str, Union[int, List[int]]]],
        batch_size: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        batches = []
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i : i + batch_size]
            tokens_batch = torch.tensor(
                [item["tokens"] for item in batch], dtype=torch.long
            )
            labels_batch = torch.tensor(
                [item["label"] for item in batch], dtype=torch.long
            )

            # Append batch as a dictionary
            batches.append({"tokens": tokens_batch, "labels": labels_batch})
        return batches


# Usage example:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dataset and tokenize it.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output tokenized dataset file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for tokenization.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length for tokenization.",
    )
    parser.add_argument(
        "--pad-token",
        type=str,
        default=None,
        help="Pad token to use for padding sequences.",
    )
    args = parser.parse_args()

    llama_api = LlamaCppAPI()
    llama_tokenizer = LlamaCppTokenizer(llama_api=llama_api)
    embedding_dataset = EmbeddingDataset(llama_tokenizer=llama_tokenizer)
    dataset = embedding_dataset.load(args.input)

    # Load the pad token ID from the tokenizer if provided, otherwise use -1 as a placeholder.
    if args.pad_token:
        pad_token_id = llama_tokenizer.encode(args.pad_token)[0]
    else:
        pad_token_id = -1

    # Set the maximum length for tokenization if provided, otherwise use the tokenizer's max embedding length.
    max_length = args.max_length

    tokenized_dataset = embedding_dataset.tokenize(dataset, max_length, pad_token_id)
    batched_dataset = embedding_dataset.batch(tokenized_dataset, args.batch_size)
    embedding_dataset.save(args.output, batched_dataset)

    print("Dataset processing completed successfully.")
