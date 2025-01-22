"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.api

Description: High-level Requests API for interacting with the LlamaCpp REST API.
"""

import html
import logging
from typing import Any, Dict, List

import requests

from llama_cpp_client.common.logger import get_logger
from llama_cpp_client.llama.request import LlamaCppRequest


class LlamaCppAPI:
    def __init__(
        self,
        request: LlamaCppRequest = None,
        top_k: int = 50,
        top_p: float = 0.90,
        min_p: float = 0.1,
        temperature: float = 0.7,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        n_predict: int = -1,
        seed: int = 1337,
        stream: bool = True,
        cache_prompt: bool = True,
        **kwargs: Any,  # Additional optional parameters
    ) -> None:
        """Initialize the API with default model parameters and request handler."""
        # NOTE: verbose is added to data which increases server verbosity.
        # Popping this is optional, but I'm leaving it for now.
        verbose = kwargs.get("verbose", False)
        log_level = logging.DEBUG if verbose else logging.INFO
        self.request = request or LlamaCppRequest(log_level=log_level)

        # Set model hyperparameters
        self.data = {
            "prompt": "",
            "messages": [],
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repeat_penalty": repeat_penalty,
            "n_predict": n_predict,
            "seed": seed,
            "stream": stream,
            "cache_prompt": cache_prompt,
        }

        # Update self.data with any additional parameters from kwargs
        self.data.update(kwargs)

        # Setup logger
        self.logger = get_logger(self.__class__.__name__, level=log_level)
        self.logger.debug("Initialized LlamaCppAPI instance.")

    def error(self, code: int, message: str, type: str) -> dict[str, Any]:
        """Return a dictionary representing an error response."""
        return {"error": {"code": code, "message": message, "type": type}}

    @property
    def health(self) -> Dict[str, Any]:
        """Check the health status of the API."""
        try:
            self.logger.debug("Fetching health status")
            return self.request.get("/health")
        except requests.exceptions.ConnectionError as e:
            self.logger.debug(f"Connection error while fetching health status: {e}")
            return self.error(500, e, "unavailable_error")

    @property
    def slots(self) -> List[Dict[str, Any]]:
        """Get the current slots processing state."""
        try:
            self.logger.debug("Fetching slot states")
            return self.request.get("/slots")
        except requests.exceptions.HTTPError as e:
            self.logger.debug("Error fetching slot states")
            return self.error(501, e, "unavailable_error")

    @property
    def models(self) -> dict[str, Any]:
        """Get the language model's file path for the given slot."""
        self.logger.debug("Fetching models list")
        return self.request.get("/v1/models")

    def get_model_path(self, slot: int = 0) -> str:
        return self.models["data"][slot]["id"]

    def get_vocab_size(self, slot: int = 0) -> int:
        """Get the language model's vocab size."""
        return self.models["data"][slot]["meta"]["n_vocab"]

    def get_context_size(self, slot: int = 0) -> int:
        """Get the language model's max context length."""
        return self.models["data"][slot]["meta"]["n_ctx_train"]

    def get_embed_size(self, slot: int = 0) -> int:
        """Get the language model's max positional embeddings."""
        return self.models["data"][slot]["meta"]["n_embd"]

    def get_prompt(self, slot: int = 0) -> str:
        """Get the system prompt for the language model in the given slot."""
        try:
            self.logger.debug(f"Fetching system prompt for slot: {slot}")
            return self.slots[slot]["prompt"]
        except KeyError:
            return self.slots["error"]["message"]

    def tokenize(
        self,
        content: str,
        add_special: bool = False,
        with_pieces: bool = False,
    ) -> List[int]:
        """Tokenizes a given text using the server's tokenize endpoint."""
        self.logger.debug(f"Tokenizing: {content}")
        data = {
            "content": content,
            "add_special": add_special,
            "with_pieces": with_pieces,
        }
        response = self.request.post("/tokenize", data=data)
        return response.get("tokens", [])

    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenizes a given sequence of token IDs using the server's detokenize endpoint."""
        self.logger.debug(f"Detokenizing: {token_ids}")
        data = {"tokens": token_ids}
        response = self.request.post("/detokenize", data=data)
        return response.get("content", "")

    def embedding(self, content: str) -> Any:
        """Get the embedding for the given input."""
        self.logger.debug(f"Fetching embedding for input: {content}")
        endpoint = "/embedding"
        data = {"input": content, "encoding_format": "float"}
        return self.request.post(endpoint, data)

    def completion(self, prompt: str) -> Any:
        """Send a completion request to the API using the given prompt."""
        self.logger.debug(f"Sending completion request with prompt: {prompt}")
        self.data["prompt"] = prompt
        self.logger.debug(f"Completion request payload: {self.data}")

        endpoint = "/completion"
        if self.data.get("stream"):
            self.logger.debug("Streaming completion request")
            return self.request.stream(endpoint=endpoint, data=self.data)
        else:
            self.logger.debug("Sending non-streaming completion request")
            return self.request.post(endpoint=endpoint, data=self.data)

    def chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """Send a ChatML-compatible chat completion request to the API."""
        self.logger.debug(f"Sending chat completion request with messages: {messages}")
        self.data["messages"] = messages

        endpoint = "/v1/chat/completions"
        if self.data.get("stream"):
            self.logger.debug("Streaming chat completion request")
            return self.request.stream(endpoint=endpoint, data=self.data)
        else:
            self.logger.debug("Sending non-streaming chat completion request")
            return self.request.post(endpoint=endpoint, data=self.data)

    def sanitize(self, text: str) -> str:
        """Escape special symbols in a given text."""
        self.logger.debug(f"Sanitizing text: {text}")
        sanitized_text = html.escape(text)
        body = []

        for symbol in sanitized_text:
            symbol = {
                "[": "\\[",  # &lbrack;
                "]": "\\]",  # &rbrack;
            }.get(symbol, symbol)
            body.append(symbol)

        final_sanitized_text = "".join(body)
        self.logger.debug(f"Sanitized text: {final_sanitized_text}")
        return final_sanitized_text


if __name__ == "__main__":
    import argparse
    import sys  # Allow streaming to stdout

    from rich import print  # Decorate output

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--assertion", help="Enable assertions", action="store_true"
    )
    parser.add_argument("-d", "--debug", help="Enable debugging", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO

    # Create an instance of LlamaCppAPI
    # llama_api = LlamaCppAPI(n_predict=45, log_level=logging.DEBUG)
    llama_api = LlamaCppAPI(n_predict=45, log_level=log_level)

    if args.assertion:
        assert (
            llama_api.sanitize("[INST] Test [/INST]") == "\\[INST\\] Test \\[/INST\\]"
        )
        assert (
            llama_api.sanitize("This is [example] text.")
            == "This is \\[example\\] text."
        )
        assert llama_api.sanitize("No brackets here!") == "No brackets here!"

    if args.debug:
        # Example: Get health status of the Llama.cpp server
        print("Health Status:", llama_api.health)
        # Example: Get slots processing state
        print("Slots State:", llama_api.slots)  # this works fine

        # Example: Get model file path for a specific slot
        slot_index = 0
        model_path = llama_api.get_model(slot=slot_index)
        print(f"Model Path for Slot {slot_index}: {model_path}")

        # Example: Get prompt for a specific slot
        prompt = llama_api.get_prompt(slot=slot_index)
        prompt = llama_api.sanitize(prompt)
        print(f"Prompt for Slot {slot_index}: {prompt}")

    # ---
    print("Running completion...")
    # ---

    # Example: Generate prediction given a prompt
    prompt = "Once upon a time"
    print(prompt, end="")

    predictions = llama_api.completion(prompt)
    # Handle the model's generated response
    content = ""
    for predicted in predictions:
        if "content" in predicted:
            token = predicted["content"]
            content += token
            # Print each token to the user
            print(token, end="")
            sys.stdout.flush()
    print()  # Add padding to the model's output

    # ---
    print("\nRunning chat completion...")
    # ---

    # Example: Generate chat completion given a sequence of messages
    messages = [
        {"role": "user", "content": "Hello! How are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking."},
        {"role": "user", "content": "Can you tell me a joke?"},
    ]
    for message in messages:
        print(f'{message["role"]}:', message["content"])

    chat_completions = llama_api.chat_completion(messages)
    # Handle the models generated response
    content = ""
    print("assistant: ", end="")
    for completed in chat_completions:
        if "content" in completed["choices"][0]["delta"]:
            # extract the token from the completed
            token = completed["choices"][0]["delta"]["content"]
            # append each chunk to the completed
            content += token
            print(token, end="")  # print the chunk out to the user
            sys.stdout.flush()  # flush the output to standard output
    print()  # add padding to models output
