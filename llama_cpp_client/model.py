"""
Copyright (C) 2024 Austin Berrio

llama_cpp_client/model.py

"Embrace the journey of discovery and evolution in the world of software development, and remember that adaptability is key to staying resilient in the face of change."
    - OpenAI's GPT-3.5
"""

import json
import os
import sys
from typing import Any, Callable

import dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

# Load environment variables from a .env file if available
if dotenv.load_dotenv(".env"):
    api_key = os.getenv("OPENAI_API_KEY")
else:
    raise ValueError("EnvironmentError: Failed to load `OPENAI_API_KEY`")

if api_key == "sk-no-key-required":
    # Correct base URL for llama.cpp
    base_url = "http://localhost:8080/v1"

# Set up the OpenAI API configuration
client = OpenAI(api_key=api_key, base_url=base_url)

# Define the conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a limerick about Python exceptions."},
]

# Request the completion with streaming
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Default to GPT-4o
        messages=messages,
        stream=True,  # Enable streaming mode
    )

    # Stream and print the responses as they arrive
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            # Stream print without newline until completion
            print(chunk_message, end="")
            sys.stdout.flush()
    print()  # add newline
except Exception as e:
    print(f"An error occurred: {e}")
