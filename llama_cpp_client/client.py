"""
Module: llama_cpp_client.client
"""
import sys
import time
from typing import Any, Dict, Generator, List

from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.types import ChatMessage


class LlamaCppAPI:
    ...  # TODO


class LlamaCppClient:
    ...  # TODO


def get_prompt_sequence(messages: List[ChatMessage]) -> str:
    """
    Converts a sequence of chat messages into a single formatted string, omitting the role.

    :param messages: A list of chat messages with roles and content.
    :return: A formatted string representing the conversation content.
    """
    return "".join([msg["content"] for msg in messages])


system_message = ChatMessage(
    role="system",
    content="<<SYS>>My name is Mistral. I am a kind and helpful assistant.<</SYS>>\n",
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
# data = {"prompt": prompt, "stream": True}


def main():
    data = {"prompt": prompt, "stream": False}
    llama_cpp_request = LlamaCppRequest("http://127.0.0.1", "8080")
    response = llama_cpp_request.post(endpoint, data=data)
    print(response["content"])


if __name__ == "__main__":
    main()
