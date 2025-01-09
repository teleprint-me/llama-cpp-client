"""
Module: llama_cpp_client.grammar

This module defines grammars using an Enum for easy reference in the project.
"""

import os
from enum import Enum


# Function to read grammar file
def read_grammar(file_path) -> str:
    try:
        with open(file_path, "r") as grammar_file:
            return grammar_file.read()
    except FileNotFoundError:
        FileNotFoundError(f"Grammar file not found at {file_path}")


# Determine the path to the grammar files within your package
current_dir = os.path.dirname(__file__)
json_gbnf = os.path.join(current_dir, "grammars", "json.gbnf")
arithmetic_gbnf = os.path.join(current_dir, "grammars", "arithmetic.gbnf")


class LlamaCppGrammar(Enum):
    """
    Enum defining grammars for use in the project.

    Attributes:
        JSON (str): JSON grammar definition.
        ARITHMETIC (str): Arithmetic grammar definition.
    """

    JSON = read_grammar(json_gbnf)
    ARITHMETIC = read_grammar(arithmetic_gbnf)
