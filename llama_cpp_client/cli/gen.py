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
        content = prompt[:]
        try:
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
        finally:
            # Save the result if output file is provided
            if output_file:
                self.save(content, output_file)
        return content

    def save(self, content: str, file_path: str) -> None:
        """Save content to a file."""
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Content saved to {file_path}")


def parse_response(response: str) -> dict:
    """
    Parse the LLM response into a structured dataset entry.
    Assumes response follows the prompt's format.

    Args:
        response (str): Generated response from the LLM.

    Returns:
        dict: Parsed entry with query, related, unrelated, and scores.
    """
    lines = response.split("\n")
    block_in = False
    block_start = "```json"
    block_end = "```"
    block = "[\n"

    for line in lines:
        if line == block_start:
            block_in = True
            continue
        if line == block_end:
            block_in = False
            block += ",\n"
            continue
        if block_in:
            block += line
    block += "\n]"
    return json.loads(block)


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
        "-n",
        "--num-entries",
        type=int,
        default=1,
        help="Number of entries to generate (for dataset generation).",
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

    # Generate dataset or single content
    if args.num_entries > 1:
        dataset = []
        for i in range(args.num_entries):
            print(f"\nGenerating entry {i + 1}/{args.num_entries}...")
            response = llama_auto.generate(prompt)
            entry = parse_response(response)
            dataset.append(entry)

        if args.output:
            llama_auto.save(json.dumps(dataset, indent=2), args.output)
    else:
        response = llama_auto.generate(prompt, output_file=args.output)
        print(f"\nModel produced {llama_auto.token_count(response)} tokens.")


if __name__ == "__main__":
    main()
