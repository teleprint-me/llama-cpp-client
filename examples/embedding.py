"""
Script: examples/embedding.py
Example of embedding generation using llama.cpp's API.
NOTE: This is not the OpenAI compatible API endpoint. See examples/oai-embedding.py for OpenAI compatible API endpoint.
"""

import requests

# Initialize parameters
url = "http://localhost:8080/embedding"  # Allows pooling set to None
headers = {"Content-Type": "application/json"}
data = {"input": ["Hello, world!", "Another example text"], "model": "my_model"}

# Send the request
response = requests.post(url, headers=headers, json=data)
embeddings = response.json()  # embeddings is a list of dict
print(embeddings)
# Extract the embeddings from the response
for result in embeddings:
    print(f"Index: {result['index']}, Embedding: {result['embedding']}")
