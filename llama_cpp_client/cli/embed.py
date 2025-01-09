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
        click.option(
            "--db-path",
            type=click.Path(exists=False, file_okay=True, dir_okay=False),
            required=True,
            help="Path to the SQLite database file.",
        ),
        click.option(
            "--verbose",
            is_flag=True,
            default=False,
            help="Enable verbose logging for debugging purposes.",
        ),
        click.option(
            "--chunk-size",
            type=int,
            default=0,
            help="Size of chunks for splitting documents. Defaults to model's embedding size.",
        ),
        click.option(
            "--batch-size",
            type=int,
            default=512,
            help="Number of tokens to process per batch. Defaults to 512.",
        ),
    ]
    for decorator in reversed(decorators):
        command_func = decorator(command_func)
    return command_func


@click.group()
def cli():
    """
    LLAMA Embedding and Query CLI.
    """
    pass


@click.command("populate")
@click.option(
    "--file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to a single file to embed.",
)
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory containing multiple text files to embed.",
)
@common_options
def populate(db_path, verbose, chunk_size, batch_size, file, directory):
    """
    Populate the LLAMA database with embeddings from a file or directory.

    Provide either --file for a single file or --directory for batch processing.
    """
    if not file and not directory:
        raise click.UsageError("You must provide either --file or --directory.")

    llama_database = LlamaCppDatabase(db_path=db_path, verbose=verbose)

    if file:
        click.echo(f"Processing file: {file}")
        llama_database.insert_embedding_from_file(
            file_path=file, chunk_size=chunk_size, batch_size=batch_size
        )
    if directory:
        click.echo(f"Processing directory: {directory}")
        llama_database.insert_embeddings_from_directory(
            dir_path=directory, chunk_size=chunk_size, batch_size=batch_size
        )


@click.command("query")
@click.argument("query", type=str)
@click.option(
    "--top-n",
    type=int,
    default=10,
    show_default=True,
    help="Number of top results to display.",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean", "manhattan"]),
    default="cosine",
    show_default=True,
    help="Similarity metric to use for ranking.",
)
@click.option(
    "--normalize",
    is_flag=True,
    default=False,
    show_default=True,
    help="Normalize similarity scores before ranking.",
)
@click.option(
    "--rerank",
    is_flag=True,
    default=False,
    show_default=True,
    help="Rerank the results using the chosen similarity metric.",
)
@common_options
def query(
    db_path,
    verbose,
    chunk_size,
    batch_size,
    query,
    top_n,
    metric,
    normalize,
    rerank,
):
    """
    Query the LLAMA database for the most similar embeddings to a given input.

    Options allow for metric selection, score normalization, and reranking.
    """
    if top_n <= 0:
        raise click.BadParameter("--top-n must be greater than 0.")

    llama_database = LlamaCppDatabase(db_path=db_path, verbose=verbose)

    # Generate the query embedding
    query_embedding = llama_database.query_embeddings(query)

    # Perform the search or rerank, based on user preference
    if rerank:
        results = llama_database.rerank_embeddings(
            query_embeddings=query_embedding,
            metric=metric,
            normalize_scores=normalize,
            top_n=top_n,
        )
    else:
        results = llama_database.search_embeddings(
            query_embeddings=query_embedding,
            metric=metric,
            normalize_scores=normalize,
            top_n=top_n,
        )

    # Display the results
    click.echo(f"\nTop {top_n} Results for Query: '{query}' (Metric: {metric})\n")
    for rank, result in enumerate(results):
        click.echo(f"Rank {rank}:")
        click.echo(f"  File: {result['file_path']}")
        click.echo(f"  Chunk ID: {result['chunk_id']}")
        click.echo(f"  Similarity Score: {result['score']:.4f}")
        click.echo(f"  Content: {result['chunk'][:200]}...\n")


# Add commands to the CLI group
cli.add_command(populate)
cli.add_command(query)

if __name__ == "__main__":
    cli()
