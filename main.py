from typing import List, Literal, TypedDict

import requests


class ChatCompletion(TypedDict):
    role: Literal["system", "assistant", "function", "user"]
    content: str


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


def get_prompt_sequence(messages: List[ChatCompletion]) -> str:
    """
    Converts a sequence of chat messages into a single formatted string.

    :param messages: A list of chat messages with roles and content.
    :return: A formatted string representing the conversation.
    """
    return "".join([f"{msg['role']}: {msg['content']}" for msg in messages])


system_message = ChatCompletion(
    role="system",
    content="<<SYS>>My name is Mistral and I am a helpful assistant<</SYS>>\n",
)
user_message = ChatCompletion(
    role="user",
    content="[INST] Hello! My name is Austin. What is your name? [/INST]\n",
)
messages = [system_message, user_message]

# Create the prompt using the get_prompt_sequence function
prompt = get_prompt_sequence(messages)

# Example usage:
endpoint = "/completion"
data = {"prompt": prompt}

response = make_request(endpoint, data=data)
print(response["content"])
