"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.request

Description: Module for handling low-level requests to the LlamaCpp REST API.
"""

import json
import logging
from typing import Any, Dict, Generator

import requests

from llama_cpp_client.common.logger import get_logger


class StreamNotAllowedError(Exception):
    def __init__(
        self, message="Streaming not allowed for this request. Set 'stream' to False."
    ):
        super().__init__(message)


class LlamaCppRequest:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1",
        port: str = "8080",
        headers: Dict[str, str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the LlamaCppRequest instance.

        :param base_url: The base URL of the server.
        :param port: The port number to use.
        :param headers: Optional headers to include in requests.
        :param log_level: The log level for this instance (e.g., logging.DEBUG).
        """
        self.base_url = f"{base_url}:{port}"
        self.headers = headers or {"Content-Type": "application/json"}
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, level=log_level)
        self.logger.debug("Initialized LlamaCppRequest instance.")

    def _handle_response(self, response: requests.Response) -> Any:
        """
        Handle the HTTP response.

        :param response: The HTTP response object.
        :return: The parsed JSON response.
        """
        self.logger.debug(f"Received response with status {response.status_code}")
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        Perform an HTTP GET request.

        :param endpoint: The API endpoint to send the GET request to.
        :param params: Optional query parameters to include in the request.
        :return: The parsed JSON response.
        """
        if params and params.get("stream", False):
            raise StreamNotAllowedError()

        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"GET request to {url} with params: {params}")
        response = requests.get(url, params=params, headers=self.headers)
        return self._handle_response(response)

    def post(self, endpoint: str, data: Dict[str, Any] = None) -> Any:
        """
        Perform an HTTP POST request.

        :param endpoint: The API endpoint to send the POST request to.
        :param data: The data to include in the request body.
        :return: The parsed JSON response.
        """
        if data and data.get("stream", False):
            raise StreamNotAllowedError()

        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"POST request to {url} with data: {data}")
        response = requests.post(url, json=data, headers=self.headers)
        return self._handle_response(response)

    def stream(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream an HTTP request.

        :param endpoint: The API endpoint to stream to.
        :param data: Data to be sent with the request (must include 'stream': True).
        :return: A generator of response data.
        """
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary containing 'stream': True.")
        if not data.get("stream", True):
            raise ValueError("Stream must be set to True for streaming requests.")

        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Streaming request to {url} with data: {data}")

        response = requests.post(url, json=data, headers=self.headers, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = line[len("data: ") :]
                if chunk == b"[DONE]":
                    self.logger.debug("Streaming complete: [DONE] signal received.")
                    break
                try:
                    decoded_chunk = json.loads(chunk)
                    self.logger.debug(f"Stream chunk received: {decoded_chunk}")
                    yield decoded_chunk
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON chunk: {chunk}")
                    raise e


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Enable debugging", action="store_true")
    parser.add_argument(
        "-p", "--prompt", help="Model input.", default="Once upon a time"
    )
    parser.add_argument(
        "-n", "--predict", help="Tokens generated.", default=-1, type=int
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO

    # Initialize the LlamaCppRequest instance
    llama_cpp_request = LlamaCppRequest(
        base_url="http://127.0.0.1", port="8080", log_level=log_level
    )

    # Define the prompt for the model
    print(args.prompt, end="")

    # Prepare data for streaming request
    data = {"prompt": args.prompt, "n_predict": args.predict, "stream": True}

    # Generate the model's response
    generator = llama_cpp_request.stream("/completion", data=data)

    # Handle the model's generated response
    content = ""
    for response in generator:
        if "content" in response:
            token = response["content"]
            content += token
            # Print each token to the user
            print(token, end="")
            sys.stdout.flush()

    # Add padding to the model's output
    print()
