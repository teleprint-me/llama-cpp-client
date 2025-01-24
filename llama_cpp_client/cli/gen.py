"""
Script: llama_cpp_client.cli.gen
Description: CLI tool for generating content or datasets using LlamaCppAuto.
"""

import argparse
import sys

from llama_cpp_client.common.args import (
    add_common_api_args,
    add_common_general_args,
    add_common_request_args,
)
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
    add_common_request_args(parser)
    add_common_api_args(parser)
    add_common_general_args(parser)
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
    llama_request = LlamaCppRequest(args.base_url, args.port)

    # Initialize core REST API
    stop = [token for token in args.stop.split(",") if token]
    llama_api = LlamaCppAPI(
        llama_request,
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
    if args.output and args.parse:
        parsed_entries = llama_auto.parse_blocks(
            response,
            args.block_start,
            args.block_end,
        )
        if parsed_entries:
            llama_auto.save(parsed_entries, args.output)
        else:
            print("Failed to parse entries.")
    elif args.output and not args.parse:
        llama_auto.save(response, args.output)
    elif not args.output and args.verbose:
        print("Did not write parsed response to output.")


if __name__ == "__main__":
    main()
