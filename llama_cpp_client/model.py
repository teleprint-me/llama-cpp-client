"""
Copyright (C) 2024 Austin Berrio

llama_cpp_client/model.py

"Embrace the journey of discovery and evolution in the world of software development, and remember that adaptability is key to staying resilient in the face of change."
    - OpenAI's GPT-3.5
"""

from openai import OpenAI

# Configure the client for llama.cpp
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Local API server for llama.cpp
    api_key="sk-no-key-required",  # No real key required for llama.cpp)
)

# Define the conversation history in the messages list
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Use a supported llama.cpp model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a limerick about Python exceptions."},
    ],
    stream=True,
)

# Print out the generated response
try:
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
