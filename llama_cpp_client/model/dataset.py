"""
Module: llama_cpp_client.model.dataset
Description: This module provides functions to load and process datasets for the Llama model.
"""

import json
import logging
from typing import Dict, List, Union

import torch

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.tokenizer import LlamaCppTokenizer


class LlamaCppDataset:
    def __init__(
        self,
        llama_api: LlamaCppAPI = None,
        llama_tokenizer: LlamaCppTokenizer = None,
        verbose: bool = False,
    ):
        self.llama_api = llama_api if llama_api else LlamaCppAPI(verbose=verbose)
        self.llama_tokenizer = (
            llama_tokenizer if llama_tokenizer else LlamaCppTokenizer(verbose=verbose)
        )
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    @property
    def api(self) -> LlamaCppAPI:
        return self.llama_api

    @property
    def tokenizer(self) -> LlamaCppTokenizer:
        return self.llama_tokenizer

    def load(self, file_path: str) -> Dict[str, List[Union[int, str]]]:
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON file: {e}")
                return {}

    def tokenize_dataset(
        self,
        dataset: Dict[str, List[Union[int, str]]],
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
                # Get the List of tensors for the current relation type
                for item in entry[relation_type]:
                    document = item["document"]
                    label = item["label"]

                    # Tokenize document
                    document_tokens = self.tokenizer.encode(
                        document, add_special_tokens=False
                    )

                    # Concatenate query and document tokens
                    tokens = query_tokens + document_tokens
                    # Truncate and pad the sequence (we can't know what the pad id is until runtime)
                    tokens = tokens[:max_length] + [pad_token_id] * max(
                        0, max_length - len(tokens)
                    )

                    # Add to tokenized data
                    tokenized_data.append({"tokens": tokens, "label": label})

        return tokenized_data

    def batch_dataset(
        self,
        tokenized_data: List[Dict[str, Union[int, List[int]]]],
        batch_size: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        batches = []
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i : i + batch_size]
            tokens_batch = torch.tensor(
                [item["tokens"] for item in batch], dtype=torch.LongTensor
            )
            labels_batch = torch.tensor(
                [item["label"] for item in batch], dtype=torch.LongTensor
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
        default=None,
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
    llama_dataset = LlamaCppDataset(
        llama_api=llama_api, llama_tokenizer=llama_tokenizer
    )
    dataset = llama_dataset.load(args.input)

    # Load the pad token ID from the tokenizer if provided, otherwise use -1 as a placeholder.
    if args.pad_token:
        pad_token_id = llama_tokenizer.encode(args.pad_token)
    else:
        pad_token_id = -1

    # Set the maximum length for tokenization if provided, otherwise use the tokenizer's max embedding length.
    if args.max_length is None:
        # NOTE: Possible alternative is .max_sequence_length()
        max_length = llama_tokenizer.max_embedding_length()
    else:
        max_length = args.max_length

    tokenized_dataset = llama_dataset.tokenize_dataset(
        dataset, max_length, pad_token_id
    )
    batched_dataset = llama_dataset.batch_dataset(tokenized_dataset, args.batch_size)
