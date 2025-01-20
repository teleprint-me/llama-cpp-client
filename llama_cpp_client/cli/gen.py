"""
Script: llama_cpp_client.cli.gen
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

        final_sanitized_text = "".join(body)
        return final_sanitized_text

    def max_tokens(self) -> int:
        return self.llama_api.get_context_size()

    def token_count(self, prompt: str) -> int:
        return len(self.llama_api.tokenize(prompt))

    def generate(self, prompt: str) -> str:
        content = prompt[:]
        try:
            generator = self.llama_api.completion(prompt)
            for response in generator:
                if "content" in response:
                    token = response["content"]
                    content += token
                    print(token, end="")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
        return content

    def save(self, content: str) -> None:
        # Save to JSON
        with open(self.file_path, "w") as f:
            f.write(content)
        print(f"Dataset saved to {self.file_path}")


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
    query = lines[0].replace("Query:", "").strip()
    related = [
        line.replace("- Related:", "").strip() for line in lines if "Related" in line
    ]
    unrelated = [
        line.replace("- Unrelated:", "").strip()
        for line in lines
        if "Unrelated" in line
    ]
    score = [0.9 for _ in related]  # Placeholder: Adjust scoring logic if needed
    return {
        "query": query,
        "related_documents": related,
        "unrelated_documents": unrelated,
        "score": score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt using raw input.")
    parser.add_argument("-f", "--prompt-file", help="Prompt using a file.")
    args = parser.parse_args()

    llama_auto = LlamaCppAuto()
    print(f"Model can use up to {llama_auto.max_tokens()} max tokens.")
    print(args.prompt, end="")
    llama_auto.generate(args.prompt)

    # generate_dataset(llama_auto, num_entries=50, output_file="semantic_dataset.json")
