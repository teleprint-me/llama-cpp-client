import sys
import time
from typing import List

import requests

from llama_cpp_client.types import ChatMessage


def make_request(endpoint, method="POST", data=None):
    """
    Send a request to the local server.

    :param endpoint: The API endpoint to send the request to.
    :param method: HTTP method to use (default is 'POST').
    :param data: Data to be sent with the request, expected to be a dictionary.
    :return: The server's response.
    """
    url = f"http://127.0.0.1:8080{endpoint}"
    headers = {"Content-Type": "application/json"}

    if method.upper() == "POST":
        response = requests.post(url, json=data, headers=headers)
    else:
        response = requests.get(url, params=data, headers=headers)

    return response.json()


def stream_request(endpoint, method="POST", data=None):
    """
    Send a request to the local server.

    :param endpoint: The API endpoint to send the request to.
    :param method: HTTP method to use (default is 'POST').
    :param data: Data to be sent with the request, expected to be a dictionary.
    :return: The server's response.
    """
    # >>> response = requests.post(url, json=data, headers=headers, stream=True)
    # >>> lines = response.iter_lines()
    # >>> for line in lines:
    # ...     print(line)
    # ...
    # b'data: {"content":"Hello","multimodal":false,"slot_id":0,"stop":false}'
    # b''
    # b'data: {"content":" Austin","multimodal":false,"slot_id":0,"stop":false}'
    # b''
    # b'data: {"content":"!","multimodal":false,"slot_id":0,"stop":false}'
    # b''
    url = f"http://127.0.0.1:8080{endpoint}"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=data, headers=headers, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = line[len("data: ") :]
            print(chunk, end=None)
            sys.stdout.flush()
    print()

    return response.json()


def get_prompt_sequence(messages: List[ChatMessage]) -> str:
    """
    Converts a sequence of chat messages into a single formatted string, omitting the role.

    :param messages: A list of chat messages with roles and content.
    :return: A formatted string representing the conversation content.
    """
    return "".join([msg["content"] for msg in messages])


system_message = ChatMessage(
    role="system",
    content="<<SYS>>My name is Mistral and I am a helpful assistant.<</SYS>>\n",
)
user_message = ChatMessage(
    role="user",
    content="[INST] Hello! My name is Austin. What is your name? [/INST]\n",
)
messages = [system_message, user_message]

# Create the prompt using the get_prompt_sequence function
prompt = get_prompt_sequence(messages)

# Example usage:
endpoint = "/completion"
data = {"prompt": prompt, "stream": True}


def main():
    response = make_request(endpoint, data=data)
    print(response["content"])


if __name__ == "__main__":
    main()
