"""
Module: llama_cpp_client.common.json
Description: This module provides utility functions for working with JSON data.
"""

import json
from typing import Any, Dict, List, Union

import numpy as np


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        Any: The loaded data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"JSON loaded from {file_path}")
    return data


def save_json(data: Union[List, Dict], file_path: str) -> None:
    """
    Dump data to a JSON file with support for NumPy types.
    Args:
        data (object): The data to be dumped.
        file_path (str): The path to the JSON file.
    Returns:
        None:
    """

    def default_serializer(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)  # Convert NumPy floats to Python floats
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)  # Convert NumPy integers to Python integers
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)
    print(f"JSON saved to {file_path}")


def parse_json(json_string: str) -> Dict[str, Any]:
    """
    Parse a JSON string into a dictionary.
    Args:
        json_string (str): The JSON string to parse.
    Returns:
        Dict[str, Any]: The JSON data as a dictionary.
    """
    return json.loads(json_string)
