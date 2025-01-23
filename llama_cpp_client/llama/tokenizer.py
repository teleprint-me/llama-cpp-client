"""
Module: llama_cpp_client.llama.tokenizer

Description:
This module provides a tokenizer class for handling text tokenization in the Llama model.
It includes methods for encoding and decoding text to and from token IDs.
"""

import logging
from typing import Dict, List, Union

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.api import LlamaCppAPI


class LlamaCppTokenizer:
    """
    Tokenizer class for LlamaCpp supported models.
    """

    def __init__(self, llama_api: LlamaCppAPI = None, verbose: bool = False):
        self.llama_api = llama_api if llama_api else LlamaCppAPI(verbose=verbose)
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)
        self.logger.debug("Tokenizer initialized")

    def max_sequence_length(self) -> int:
        """
        Returns the maximum sequence length supported by the model.
        """
        length = self.llama_api.get_context_size()
        self.logger.debug(f"Maximum sequence length: {length}")
        return length

    def max_embedding_length(self) -> int:
        """
        Returns the maximum embedding length supported by the model.
        """
        length = self.llama_api.get_embed_size()
        self.logger.debug(f"Maximum embedding length: {length}")
        return length

    def encode(
        self,
        prompt: str,
        add_special_tokens: bool = False,
        with_pieces: bool = False,
    ) -> Union[List[int], List[Dict[str, Union[int, str]]]]:
        """
        Tokenizes the given prompt using the Llama model's tokenizer. Optionally, adds special tokens and returns pieces.
        Args:
            prompt (str): The input prompt to tokenize.
            add_special_tokens (bool): Whether to add special tokens to the prompt.
            with_pieces (bool): Whether to return pieces instead of token IDs.
        Returns:
            Union[List[int], List[Dict[str, Union[int, str]]]]: A list of token IDs or a list of dictionaries with 'id' and 'piece' keys.
        """
        pieces = self.llama_api.tokenize(prompt, add_special_tokens, with_pieces)
        self.logger.debug(f"Tokenized prompt: {pieces}")
        return pieces

    def decode(self, pieces: Union[List[int], List[Dict[str, Union[int, str]]]]) -> str:
        """
        Decodes a list of token IDs back into a string using the Llama model's tokenizer.
        Args:
            pieces (List[int]): The list of token IDs or dictionaries representing token IDs and pieces to decode.
        Returns:
            str: The decoded string.
        """
        if isinstance(pieces, list) and isinstance(pieces[0], dict):
            self.logger.debug("Decoding pieces with 'id' and 'piece' keys")
            # If pieces is a list of dictionaries, extract the 'id' values
            pieces = [piece["id"] for piece in pieces]
        # Decode the token IDs back into a string using the Llama model's tokenizer
        tokens = self.llama_api.detokenize(pieces)
        self.logger.debug(f"Decoded tokens: {tokens}")
        return tokens
