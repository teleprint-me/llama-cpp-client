"""
Module: llama_cpp_client.tokenizer
"""

from typing import List

from llama_cpp_client.request import LlamaCppRequest


class LlamaCppTokenizer:
    def __init__(self, model_request: LlamaCppRequest):
        """
        Initializes the LlamaCppTokenizer with the server URL.

        :param server_url: The base URL of the server where the tokenize and detokenize endpoints are available.
        """
        self.model_request = model_request or LlamaCppRequest()

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a given text using the server's tokenize endpoint.

        :param text: The text to tokenize.
        :return: A list of token IDs.
        """
        payload = {"content": text}
        model_response = self.model_request.post("/tokenize", data=payload)
        return model_response.get("tokens", [])

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a given sequence of token IDs using the server's detokenize endpoint.

        :param tokens: The list of token IDs to detokenize.
        :return: The detokenized text.
        """
        payload = {"tokens": tokens}
        response = self.model_request.post("/detokenize", data=payload)
        return response.get("content", "")


# Example usage:
if __name__ == "__main__":
    model_request = LlamaCppRequest(base_url="http://127.0.0.1", port="8080")
    tokenizer = LlamaCppTokenizer(model_request=model_request)
    text = "<|system|>\nMy name is StableLM. I am a helpful assistant.<|endoftext|>\n<|user|>\nHello! My name is Austin! What is your name?<|endoftext|>\n<|assistant|>\n"
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    detokenized_text = tokenizer.detokenize(tokens)
    print(f"Detokenized Text: {detokenized_text}")
