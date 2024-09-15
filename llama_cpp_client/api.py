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
        **kwargs,  # Additional optional parameters
    ) -> None:
        # manage llama.cpp client instances
        self.request = request or LlamaCppRequest()

        # set model hyper parameters
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

        # Update self.data with any additional parameters provided via kwargs
        self.data.update(kwargs)

        # Debugging is disabled by default
        log_level = kwargs.get("log_level", logging.INFO)
        self.logger = get_default_logger("LlamaCppAPI", level=log_level)

    @property
    def health(self) -> Dict[str, Any]:
        try:
            self.logger.debug("Fetching health status")
            return self.request.get("/health")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"ConnectionError: {e}")
            return {"status": "error", "message": e}

    @property
    def slots(self) -> List[Dict[str, Any]]:
        """Get the current slots processing state."""
        self.logger.debug("Fetching slot states")
        return self.request.get("/slots")

    def get_ctx_size(self, slot: int = 0) -> int:
        """Get the language models max positional embeddings"""
        # NOTE: Return -1 to indicated an error occurred
        return self.slots[slot].get("n_ctx", -1)

    def get_model(self, slot: int = 0) -> str:
        """Get the language models file path"""
        # NOTE: A slot is allocated to a individual user
        return self.slots[slot].get("model", "")

    def get_prompt(self, slot: int = 0) -> str:
        """Get the language models system prompt"""
        return self.slots[slot].get("prompt", "")

    def completion(self, prompt: str) -> Any:
        """Get a prediction given a prompt"""
        self.logger.debug(f"Sending completion request with prompt: {prompt}")
        endpoint = "/completion"
        self.data["prompt"] = prompt
        self.logger.debug(f"Completion request data: {self.data}")

        if self.data.get("stream"):
            return self.request.stream(endpoint=endpoint, data=self.data)
        else:
            return self.request.post(endpoint=endpoint, data=self.data)

    def chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """Get a OpenAI ChatML compatible prediction given a sequence of messages"""
        endpoint = "/v1/chat/completions"
        self.data["messages"] = messages

        if self.data.get("stream"):
            return self.request.stream(endpoint=endpoint, data=self.data)
        if not self.data.get("stream"):
            return self.request.post(endpoint=endpoint, data=self.data)

    @staticmethod
    def sanitize(text: str) -> str:
        """Escape special symbols from a given body of text."""
        # NOTE: This is a monkey patch and requires refining.
        body = ""
        for symbol in text:
            if symbol == "[":
                symbol = "\\["
            if symbol == "]":
                symbol = "\\]"
            body += symbol
        return html.escape(body)


if __name__ == "__main__":
    import sys  # Allow streaming to stdout

    from rich import print  # Decorate output

    # Create an instance of LlamaCppAPI
    llama_api = LlamaCppAPI(n_predict=45, log_level=logging.DEBUG)

    # Example: Get health status of the Llama.cpp server
    health_status = llama_api.health
    print("Health Status:", health_status)

    # Example: Get slots processing state
    slots_state = llama_api.slots
    # using json.dumps() seems to be causing conflicts with rich.print()
    print("Slots State:", slots_state)  # this works fine

    # Example: Get model file path for a specific slot
    slot_index = 0
    model_path = llama_api.get_model(slot=slot_index)
    print(f"Model Path for Slot {slot_index}: {model_path}")

    assert llama_api.sanitize("[INST] Test [/INST]") == "\\[INST\\] Test \\[/INST\\]"
    assert (
        llama_api.sanitize("This is [example] text.") == "This is \\[example\\] text."
    )
    assert llama_api.sanitize("No brackets here!") == "No brackets here!"

    # Example: Get prompt for a specific slot
    prompt = llama_api.get_prompt(slot=slot_index)
    prompt = llama_api.sanitize(prompt)
    print(f"Prompt for Slot {slot_index}: {prompt}")

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
