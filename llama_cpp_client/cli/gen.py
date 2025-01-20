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

    def generate(self, prompt: str, output_file: str = None) -> str:
        if self.token_count(prompt) > self.max_prompt():
            raise ValueError(f"Prompt exceeds the token limit: {self.max_prompt()}.")
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

    def save_text(self, content: str, file_path: str) -> None:
        """Save content to a text file."""
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Content saved to {file_path}")

    def save_json(self, data: object, file_path: str) -> None:
        """Dump data to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON saved to {file_path}")


def parse_json(response: str) -> list[dict]:
    """
    Parse the LLM response into structured dataset entries.
    Assumes response contains multiple JSON blocks, each delimited by ```json and ```.

    Args:
        response (str): Generated response from the LLM.

    Returns:
        list[dict]: A list of parsed dataset entries.
    """
    lines = response.split("\n")
    block_in = False
    block_start = "```json"
    block_end = "```"
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
    parser.add_argument("-p", "--prompt", help="Prompt text to generate content.")
    parser.add_argument(
        "-f", "--prompt-file", help="Path to a file containing the prompt."
    )
    parser.add_argument("-o", "--output", help="File to save generated content.")
    parser.add_argument(
        "-j",
        "--parse-json",
        action="store_true",
        help="Parse generated entries for dataset generation.",
    )
    parser.add_argument(
        "-s",
        "--sanitize",
        action="store_true",
        help="Sanitize prompt.",
    )
    args = parser.parse_args()

    # Initialize LlamaCppAuto
    llama_auto = LlamaCppAuto()

    # Display model info
    print(f"Model supports up to {llama_auto.max_tokens()} tokens.")

    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: Please provide a prompt or a prompt file.")
        sys.exit(1)

    if args.sanitize:
        prompt = llama_auto.sanitize(prompt)

    response = llama_auto.generate(prompt, output_file=args.output)
    print(f"\nModel produced {llama_auto.token_count(response)} tokens.")

    # Generate dataset or single content
    if args.parse_json:
        parsed_entries = parse_json(response)
        if parsed_entries:
            for entry in parsed_entries:
                print(f"Parsed Entry: {entry}")
            if args.output:
                llama_auto.save_json(parsed_entries, args.output)
        else:
            print("No valid entries were parsed from the response.")


if __name__ == "__main__":
    main()
