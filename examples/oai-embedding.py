"""
Script: examples/oai-embedding.py
Example of embedding generation using llama.cpp's OpenAI compatible API.
NOTE: The `--pooling mean` CLI option is required by the llama.cpp server. Otherwise the server will respond with a 500 internal server error.
"""

import requests

# Set the OpenAI-compatible endpoint
url = "http://localhost:8080/v1/embeddings"
headers = {"Content-Type": "application/json"}

# Input for embedding generation
data = {"input": ["Hello, world!", "Another example text"], "model": "my_model"}

# Send the request
response = requests.post(url, headers=headers, json=data)

# Handle response
if response.status_code == 200:
    embeddings = response.json()  # embeddings is a dictionary
    print(embeddings)
    for item in embeddings["data"]:
        print(f"Index: {item['index']}, Embedding: {item['embedding']}")
else:
    print(f"Error: {response.json()}")
