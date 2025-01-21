"""
Script: llama_cpp_client.cli.gen
Description: CLI tool for generating content or datasets using LlamaCppAuto.
"""

import argparse
import sys

from llama_cpp_client.llama.api import LlamaCppAPI
from llama_cpp_client.llama.auto import LlamaCppAuto
from llama_cpp_client.llama.request import LlamaCppRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate content or datasets using LlamaCppAuto."
    )
    parser.add_argument("-p", "--prompt", help="Model instruction to generate output.")
    parser.add_argument("-i", "--input", help="File to read an instruction from.")
    parser.add_argument("-o", "--output", help="File to save model outputs to.")
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize prompt. (Default: False)",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse model outputs for dataset generation. (Default: False)",
    )
    parser.add_argument(
        "--block-start",
        default="```",
        help="The start of a block to parse. (Default: ```)",
    )
    parser.add_argument(
        "--block-end",
        default="```",
        help="The end of a block to parse. (Default: ```)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1",
        help="The servers url (default: http://127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8080",
        help="The servers port (default: 8080)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Limit output tokens to top-k most likely (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Only consider tokens with prob greater than top-p (default: 0.9)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="Minimum token probability (default: 0.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for output randomness (default: 0.7)",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.0,
        help="Penalty for repeating tokens (default: 1.0, no effect)",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=-1,
        help="The number of tokens to predict (default: -1, inf)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Initial value for random number generator (default: -1)",
    )
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        help="Reuse cached prompts to speed up processing (default: False)",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="",
        help="List of stop tokens to ignore (default: empty string; use comma delimited list, no spaces).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbosity. (Default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model instructions
    if args.input:
        with open(args.input, "r") as f:
            args.prompt = f.read()

    if not args.prompt:
        print("Error: Please provide a prompt or a prompt file.")
        sys.exit(1)

    # Initialize core requests
    llama_request = LlamaCppRequest(base_url=args.base_url, port=args.port)

    # Initialize core REST API
    stop = [token for token in args.stop.split(",") if token]
    llama_api = LlamaCppAPI(
        request=llama_request,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        n_predict=args.n_predict,
        seed=args.seed,
        cache_prompt=args.cache_prompt,
        stop=stop,
    )

    # Initialize autonomous behavior
    llama_auto = LlamaCppAuto(llama_api=llama_api)

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
