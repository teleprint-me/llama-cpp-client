"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.request
"""

import json
from typing import Any, Dict, Generator

import requests


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
    ):
        """
        Initialize the LlamaCppRequest instance.

        :param base_url: The base URL of the server.
        :param port: The port number to use.
        :param headers: Optional headers to include in requests.
        """
        self.base_url = f"{base_url}:{port}"
        self.headers = headers or {"Content-Type": "application/json"}

    def _handle_response(self, response: requests.Response) -> Any:
        """
        Handle the HTTP response.

        :param response: The HTTP response object.
        :return: The parsed JSON response.
        """
        # Raise HTTP error if response status code is not OK (2xx)
        response.raise_for_status()
        return response.json()

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        Perform an HTTP GET request.

        :param endpoint: The API endpoint to send the GET request to.
        :param params: Optional query parameters to include in the request.
        :return: The parsed JSON response.
        """
        # NOTE: Unsure of whether to use params or data here.
        # Intuition is telling me it should probably be data.
        if params and params.get("stream", False):
            raise StreamNotAllowedError()

        url = f"{self.base_url}{endpoint}"
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
        response = requests.post(url, json=data, headers=self.headers)
        return self._handle_response(response)

    def stream(
        self, endpoint: str, data: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream an HTTP request.

        :param endpoint: The API endpoint to stream to.
        :param data: Data to be sent with the request.
        :return: A generator of response data.
        """
        if not data.get("stream", True):
            raise ValueError("Stream must be set to True for streaming requests.")

        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data, headers=self.headers, stream=True)

        for line in response.iter_lines():
            if line:
                chunk = line[len("data: ") :]
                yield json.loads(chunk)  # convert extracted chunk to dict


if __name__ == "__main__":
    import sys

    # Initialize the LlamaCppRequest instance
    llama_cpp_request = LlamaCppRequest(base_url="http://127.0.0.1", port="8080")

    # Define the prompt for the model
    prompt = "Once upon a time"
    print(prompt, end="")

    # Prepare data for streaming request
    data = {"prompt": prompt, "stream": True}

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
