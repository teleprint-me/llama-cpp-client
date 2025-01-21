"""
Script: llama_cpp_client.cli.gen
Description: CLI tool for generating content or datasets using LlamaCppAPI.
"""

import argparse
import html
import json
import sys

from llama_cpp_client.llama.api import LlamaCppAPI


class LlamaCppAuto:
    def __init__(self, file_path: str = None, llama_api: LlamaCppAPI = None):
        self.file_path = file_path
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate content or datasets using LlamaCppAPI."
    )
    parser.add_argument("-p", "--prompt", help="Model instruction to generate output.")
    parser.add_argument("-i", "--input", help="File to read an instruction from.")
    parser.add_argument("-o", "--output", help="File to save model outputs to.")
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize prompt.",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse model outputs for dataset generation.",
    )
    parser.add_argument(
        "--block-start",
        default="```",
        help="The start of a block to parse.",
    )
    parser.add_argument(
        "--block-end",
        default="```",
        help="The end of a block to parse.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbosity.",
    )
    args = parser.parse_args()

    # Load model instructions
    if args.input:
        with open(args.input, "r") as f:
            args.prompt = f.read()

    if not args.prompt:
        print("Error: Please provide a prompt or a prompt file.")
        sys.exit(1)

    # Initialize LlamaCppAuto
    llama_auto = LlamaCppAuto()

    if args.sanitize:
        args.prompt = llama_auto.sanitize(args.prompt)

    # Display model info
    if args.verbose:
        print(f"Model supports up to {llama_auto.max_tokens()} tokens.")
        print(f"Prompt can use {llama_auto.max_prompt()} tokens.")
        if args.sanitize:
            print("Sanitized", end=" ")
        print(f"Prompt: {args.prompt}")

    response = llama_auto.generate(args.prompt)

    if args.verbose:
        print(f"\nModel produced {llama_auto.token_count(response)} tokens.")

    # Generate dataset or single content
    if args.parse:
        parsed_entries = llama_auto.parse_blocks(
            response,
            args.block_start,
            args.block_end,
        )
        if args.output and parsed_entries:
            llama_auto.save(parsed_entries, args.output)
    if not args.parse and args.verbose:
        print("Did not write parsed response to output.")


if __name__ == "__main__":
    main()
