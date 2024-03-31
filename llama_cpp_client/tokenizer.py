"""
Module: llama_cpp_client.tokenizer
"""

from typing import List, Optional

from llama_cpp_client.request import LlamaCppRequest


class LlamaCppTokenizer:
    def __init__(self, request: Optional[LlamaCppRequest] = None):
        """
        Initializes the LlamaCppTokenizer with a LlamaCppRequest instance.

        :param server_url: The base URL of the server where the tokenize and detokenize endpoints are available.
        """
        self.llama_cpp_request = request or LlamaCppRequest()

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a given text using the server's tokenize endpoint.

        :param text: The text to tokenize.
        :return: A list of token IDs.
        """
        payload = {"content": text}
        llama_cpp_response = self.llama_cpp_request.post("/tokenize", data=payload)
        return llama_cpp_response.get("tokens", [])

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a given sequence of token IDs using the server's detokenize endpoint.

        :param tokens: The list of token IDs to detokenize.
        :return: The detokenized text.
        """
        payload = {"tokens": tokens}
        llama_cpp_response = self.llama_cpp_request.post("/detokenize", data=payload)
        return llama_cpp_response.get("content", "")


if __name__ == "__main__":
    # Initialize the LlamaCppRequest instance
    llama_cpp_request = LlamaCppRequest(base_url="http://127.0.0.1", port="8080")

    # Initialize the LlamaCppTokenizer instance
    tokenizer = LlamaCppTokenizer(llama_cpp_request=llama_cpp_request)

    # Define the text to tokenize
    text = "<|system|>\nMy name is StableLM. I am a helpful assistant.<|endoftext|>\n<|user|>\nHello! My name is Austin! What is your name?<|endoftext|>\n<|assistant|>\n"

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # Detokenize the tokens
    detokenized_text = tokenizer.detokenize(tokens)
    print(f"Detokenized Text: {detokenized_text}")
