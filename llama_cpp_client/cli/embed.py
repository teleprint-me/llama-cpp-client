"""
Script: llama_cpp_client.cli.embed

Description: Script to add file and directory embeddings to the LLAMA database.
"""

import click

from llama_cpp_client.llama.embedding import LlamaCppDatabase


def common_options(command_func):
    """
    Decorator to add common options to a command.
    """
    decorators = [
        click.option("--db-path", type=click.Path(exists=True), required=True),
        click.option("--verbose", is_flag=False),
        click.option("--chunk_size", type=int, default=0),
        click.option("--batch_size", type=int, default=512),
    ]
    for decorator in reversed(decorators):
        command_func = decorator(command_func)
    return command_func


@click.group()
def cli():
    """
    Script: llama_cpp_client.cli.embed
    """
    pass


@click.command("populate")
@click.option("--input", type=click.Path(exists=True), required=True)
@click.option("--directory", type=click.Path(exists=True), required=True)
@common_options
def populate(db_path, verbose, chunk_size, batch_size, input, directory):
    """
    Populate the LLAMA database with embeddings from a file or directory.
    """
    llama_database = LlamaCppDatabase(db_path=db_path, verbose=verbose)


# Options for the query command
# --query <string>
# --top-n <int>
@click.command("query")
@click.argument("--input", type=str)
@click.option("--top-n", type=int, default=10)
@common_options
def query(query, top_n):
    pass  # TODO
