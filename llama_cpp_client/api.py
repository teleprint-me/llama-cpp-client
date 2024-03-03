"""
Module: llama_cpp_client.api
"""

from typing import Any, Dict, List

from llama_cpp_client.history import LlamaCppHistory
from llama_cpp_client.request import LlamaCppRequest
from llama_cpp_client.tokenizer import LlamaCppTokenizer


class LlamaCppAPI:
    def __init__(
        self,
        request: LlamaCppRequest = None,
        tokenizer: LlamaCppTokenizer = None,
        history: LlamaCppHistory = None,
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
        self.tokenizer = tokenizer or LlamaCppTokenizer(self.request)
        self.history = history or LlamaCppHistory("default")

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

    @property
    def health(self) -> Dict[str, Any]:
        return self.request.get("/health")

    @property
    def slots(self) -> List[Dict[str, Any]]:
        """Get the current slots processing state."""
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
        endpoint = "/completion"
        self.data["prompt"] = prompt

        # TODO: Manage completion history

        if self.data.get("stream"):
            yield self.request.stream(endpoint=endpoint, data=self.data)
        if not self.data.get("stream"):
            return self.request.post(endpoint=endpoint, data=self.data)

    def chat_completion(self, content: str) -> Any:
        """Get a OpenAI ChatML compatible prediction given a sequence of messages"""
        endpoint = "/v1/chat/completions"
        message = {"role": "user", "content": content}
        self.history.append(message)
        self.data["messages"] = self.history.messages

        if self.data.get("stream"):
            yield self.request.stream(endpoint=endpoint, data=self.data)
        if not self.data.get("stream"):
            return self.request.post(endpoint=endpoint, data=self.data)
