"""
Copyright © 2023 Austin Berrio

Module: llama_cpp_client.llama.auto

Description: This module offers an automated tool for generating content or datasets
using the LlamaCppAPI. It allows parsing a large language model's output to parse code
blocks and save the outputs to a file.
"""

import html
import json
import sys

from llama_cpp_client.llama.api import LlamaCppAPI


class LlamaCppAuto:
    def __init__(self, llama_api: LlamaCppAPI = None):
        self.llama_api = llama_api or LlamaCppAPI()

    def sanitize(self, prompt: str) -> str:
        """Escape special symbols in a given text."""
        sanitized_text = html.escape(prompt)
        body = []

        for symbol in sanitized_text:
            symbol = {
                "[": "\\[",  # &lbrack;
                "]": "\\]",  # &rbrack;
            }.get(symbol, symbol)
            body.append(symbol)

        return "".join(body)

    def max_tokens(self) -> int:
        return self.llama_api.get_context_size()

    def max_prompt(self, limit: int = 4) -> int:
        # NOTE: Prompt should not consume more than 25% of the context window.
        limit = limit if limit > 0 else 4
        return self.max_tokens() / limit

    def token_count(self, prompt: str) -> int:
        return len(self.llama_api.tokenize(prompt))

    def generate(self, prompt: str, limit: int = 4) -> str:
        if self.token_count(prompt) > self.max_prompt(limit):
            raise ValueError(
                f"Prompt exceeds token limit: {self.token_count(prompt)} > {self.max_prompt()}."
            )
        try:
            content = ""
            generator = self.llama_api.completion(prompt)
            for response in generator:
                if "content" in response:
                    token = response["content"]
                    content += token
                    print(token, end="")
                    sys.stdout.flush()
            print()  # Add a new line after streaming output
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
        # NOTE: Do not save the file here. Do it externally. It will raise an exception otherwise.
        return content

    def save_text(self, data: str, file_path: str) -> None:
        """Save data to a text file."""
        with open(file_path, "w") as f:
            f.write(data)
        print(f"Content saved to {file_path}")

    def save_json(self, data: object, file_path: str) -> None:
        """Dump data to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON saved to {file_path}")

    def save(self, data: object, file_path: str) -> None:
        ext = file_path.split(".")[-1]
        if ext == "json":
            self.save_json(data, file_path)
            print(f"Save {file_path} as json.")
        # if ext == special_format: ...
        else:
            self.save_text(data, file_path)
            print(f"Save {file_path} as plaintext.")

    def parse_blocks(
        self,
        response: str,
        block_start: str = "```",
        block_end: str = "```",
    ) -> list[dict]:
        """
        Parse the LLM response into structured dataset entries.
        Assumes response contains multiple code blocks, each delimited by ``` and ```.

        Args:
            response (str): Generated response from the LLM.

        Returns:
            list[dict]: A list of parsed dataset entries.
        """
        lines = response.split("\n")
        block_in = False
        block_start = block_start
        block_end = block_end
        blocks = []
        current_block = ""

        for line in lines:
            if line.strip() == block_start:
                block_in = True
                current_block = ""  # Start a new block
                continue
            if line.strip() == block_end:
                block_in = False
                try:
                    # Parse the current block as JSON
                    blocks.append(json.loads(current_block))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON block: {e}")
                continue
            if block_in:
                current_block += line.strip()

        return blocks
