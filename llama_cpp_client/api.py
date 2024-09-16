"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.api
"""

import html
import logging
from typing import Any, Dict, List

import requests

from llama_cpp_client.logger import get_default_logger
from llama_cpp_client.request import LlamaCppRequest


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
        log_level = kwargs.get("log_level", logging.INFO)
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
        self.logger = get_default_logger("LlamaCppAPI", level=log_level)
        self.logger.debug("Initialized LlamaCppAPI instance.")

    @property
    def health(self) -> Dict[str, Any]:
        """Check the health status of the API."""
        try:
            self.logger.debug("Fetching health status")
            return self.request.get("/health")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error while fetching health status: {e}")
            return {"status": "error", "message": str(e)}

    @property
    def slots(self) -> List[Dict[str, Any]]:
        """Get the current slots processing state."""
        self.logger.debug("Fetching slot states")
        return self.request.get("/slots")

    def get_ctx_size(self, slot: int = 0) -> int:
        """Get the language model's max positional embeddings."""
        self.logger.debug(f"Fetching context size for slot: {slot}")
        return self.slots[slot].get("n_ctx", -1)

    def get_model(self, slot: int = 0) -> str:
        """Get the language model's file path for the given slot."""
        self.logger.debug(f"Fetching model for slot: {slot}")
        return self.slots[slot].get("model", "")

    def get_prompt(self, slot: int = 0) -> str:
        """Get the system prompt for the language model in the given slot."""
        self.logger.debug(f"Fetching system prompt for slot: {slot}")
        return self.slots[slot].get("prompt", "")

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
