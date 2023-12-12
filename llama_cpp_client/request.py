"""
Module: llama_cpp_client.request
"""
import json
from typing import Any, Dict, Generator

import requests


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
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, json=data, headers=self.headers, stream=True)

        for line in response.iter_lines():
            if line:
                chunk = line[len("data: ") :]
                yield json.loads(chunk)  # convert extracted chunk to dict
