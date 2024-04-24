"""
Module: llama_cpp_client.function.file
"""

import pathlib


def read_file(
    filepath: str | pathlib.Path,
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """
    Reads the contents of a file and returns a substring based on optional line numbers.

    Args:
        filepath (str or pathlib.Path): The required filepath to read from.
        start_line (int): Optional starting line number to extract from. Default is 0.
        end_line (int): Optional ending line number to extract up to. If None, it defaults to the last line in the file.

    Returns:
        str: A string containing the substring of file content based on given line numbers.

    Raises:
        FileNotFoundError: If the provided filepath does not exist.
        ValueError: If invalid or inconsistent start and end line numbers are provided.
    """

    if isinstance(filepath, str):
        path = pathlib.Path(filepath)

    with open(path, "r") as f:
        lines = f.readlines()

    if end_line == 0:
        end_line = len(lines) - 1

    if start_line < 0 or end_line <= start_line or end_line > len(lines):
        raise ValueError("Invalid or inconsistent starting and/or ending line numbers.")

    substring = "".join(
        [lines[i] for i in range(start_line, min(len(lines), end_line))]
    )

    return substring
