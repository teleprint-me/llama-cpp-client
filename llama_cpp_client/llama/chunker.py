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

    def __init__(self, api: LlamaCppAPI = None, verbose: bool = False) -> None:
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

    @property
    def embed_size(self) -> int:
        """Return the size of the embedding."""
        return self.api.get_embed_size()

    @property
    def special_token_count(self) -> int:
        """Return the number of special tokens."""
        # Use a dummy input to count special tokens
        return len(self.api.tokenize("", add_special=True, with_pieces=False))

    def normalize_text(
        self,
        content: str,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        preserve_structure: bool = True,
    ) -> str:
        """
        Normalize text by removing punctuation and converting to lowercase.
        Args:
            content (str): The text to be normalized.
            lowercase (bool): Whether to convert text to lowercase.
            remove_punctuation (bool): Whether to remove punctuation.
            preserve_structure (bool): Whether to preserve basic punctuation for structure.
        Returns:
            str: The normalized text.
        Raises:
            ValueError: If the content is not a string.
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")

        normalized = content.strip()
        if not normalized:
            self.logger.debug("Content is empty; skipping normalization.")
            return ""

        if lowercase:
            normalized = normalized.lower()

        if remove_punctuation:
            if preserve_structure:
                normalized = re.sub(r"[^\w\s.,?!'\"()]", "", normalized)
            else:
                normalized = re.sub(r"[^\w\s]", "", normalized)

        if self.verbose:
            self.logger.debug("Text normalization completed.")

        return normalized

    def chunk_text(
        self,
        content: str,
        batch_size: int = 512,
        chunk_size: int = 256,
        overlap: int = 0,
    ) -> List[str]:
        """
        Split text into chunks compatible with the model's embedding size and batch size.
        Args:
            content (str): The text to be chunked.
            batch_size (int): Physical batch size (defaults to 512).
            chunk_size (int): Maximum number of tokens per chunk (defaults to batch size - special tokens).
            overlap (int): Number of tokens to overlap between chunks (defaults to 0).
        Returns:
            List[str]: List of text chunks.
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")
        if not content:
            raise ValueError("Content cannot be empty.")

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
        tokens = self.api.tokenize(content, add_special=False, with_pieces=False)
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.api.detokenize(chunk_tokens)
            chunks.append(chunk_text)
            if self.verbose:
                self.logger.debug(
                    f"Chunk {i // (chunk_size - overlap)}: {len(chunk_tokens)} tokens."
                )

        if not chunks and content.strip():
            chunks.append(content.strip())
            self.logger.debug(
                "Content too short for chunking; returned as single chunk."
            )

        self.logger.debug(f"Generated {len(chunks)} chunks.")
        return chunks


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk a text into smaller pieces.")
    parser.add_argument("--text-file", type=str, help="Path to the text file to chunk.")
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (Default: False).",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger(name="chunker", level=log_level)

    with open(args.text_file, "r", encoding="utf-8") as file:
        text = file.read()

    chunker = LlamaCppChunker(verbose=args.verbose)
    chunks = chunker.chunk_text(text, args.batch_size, args.chunk_size, args.overlap)
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: {chunk}")
