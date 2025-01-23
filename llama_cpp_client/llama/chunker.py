"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.chunker

Description: A class for processing and chunking file contents using the LlamaCppAPI.
"""

import logging
import re
from typing import List

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI


class LlamaCppChunker:
    """Class for processing and chunking file contents."""

    def __init__(
        self, file_path: str, api: LlamaCppAPI = None, verbose: bool = False
    ) -> None:
        """
        Initialize the FileChunker with the API instance and file path.

        Args:
            api (LlamaCppAPI): Instance of the LlamaCppAPI.
            file_path (str): Path to the file to be processed.
            verbose (bool): Whether to enable verbose logging.
        """
        self.api = api if api is not None else LlamaCppAPI()
        self.verbose = verbose
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )
        self.file_contents = ""
        self.read_file_contents(file_path)

    @property
    def embed_size(self) -> int:
        """Return the size of the embedding."""
        return self.api.get_embed_size()

    @property
    def special_token_count(self) -> int:
        """Return the number of special tokens."""
        # Use a dummy input to count special tokens
        return len(self.api.tokenize("", add_special=True, with_pieces=False))

    def read_file_contents(self, file_path: str) -> None:
        """Read a file and store its contents."""
        try:
            with open(file_path, "r") as file:
                self.file_contents = file.read()
            if self.verbose:
                self.logger.debug(f"Read file contents from: {file_path}")
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except IOError as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

    def normalize_text(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        preserve_structure: bool = True,
    ) -> None:
        """
        Normalize text by removing punctuation and converting to lowercase.

        Args:
            lowercase (bool): Whether to convert text to lowercase.
            remove_punctuation (bool): Whether to remove punctuation.
            preserve_structure (bool): Whether to preserve basic punctuation for structure.
        """
        if not self.file_contents.strip():
            self.logger.debug("File is empty; skipping normalization.")
            return

        normalized = self.file_contents.strip()
        if lowercase:
            normalized = normalized.lower()
        if remove_punctuation:
            if preserve_structure:
                normalized = re.sub(r"[^\w\s.,?!'\"()]", "", normalized)
            else:
                normalized = re.sub(r"[^\w\s]", "", normalized)
        self.file_contents = normalized
        if self.verbose:
            self.logger.debug("Text normalization completed.")

    def chunk_text_with_model(
        self,
        chunk_size: int = 0,
        overlap: int = 0,
        batch_size: int = 512,
    ) -> List[str]:
        """
        Split text into chunks compatible with the model's embedding size and batch size.

        Args:
            chunk_size (int): Maximum number of tokens per chunk (defaults to batch size - special tokens).
            overlap (int): Number of tokens to overlap between chunks.
            batch_size (int): Physical batch size (defaults to 512).

        Returns:
            List[str]: List of text chunks.
        """
        # Determine the special token overhead
        max_tokens = batch_size - self.special_token_count
        if not (0 < chunk_size < max_tokens):
            self.logger.debug(
                f"Chunk size adjusted to {max_tokens} to fit within batch constraints."
            )
            chunk_size = max_tokens
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        chunks = []
        tokens = self.api.tokenize(
            self.file_contents, add_special=False, with_pieces=False
        )
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.api.detokenize(chunk_tokens)
            chunks.append(chunk_text)
            if self.verbose:
                self.logger.debug(
                    f"Chunk {i // (chunk_size - overlap)}: {len(chunk_tokens)} tokens."
                )

        if not chunks and self.file_contents.strip():
            chunks.append(self.file_contents.strip())
            self.logger.debug(
                "Content too short for chunking; returned as single chunk."
            )

        self.logger.debug(f"Generated {len(chunks)} chunks.")
        return chunks
