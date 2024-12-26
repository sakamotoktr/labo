import uuid
from typing import Annotated, List, Optional

import questionary
import typer
from labo import create_client
from labo.data_sources.connectors import DirectoryConnector

app = typer.Typer()

# Default file extensions for the data source files
default_extensions = ".txt,.md,.pdf"


def select_embedding_model(embedding_configs):
    """
    Select an embedding model from the available options.
    If there's only one option, return that one.
    If there are multiple options, prompt the user to choose one.
    Raises a ValueError if no embedding models are found.
    """
    embedding_options = [embedding_config.embedding_model for embedding_config in embedding_configs]
    if len(embedding_options) == 0:
        raise ValueError("No embedding models found. Please enable a provider.")
    elif len(embedding_options) == 1:
        return embedding_options[0]
    else:
        embedding_choices = [
            questionary.Choice(title=embedding_config.pretty_print(), value=embedding_config) for embedding_config in embedding_configs
        ]
        selected_embedding_config = questionary.select("Select embedding model:", choices=embedding_choices).ask()
        return selected_embedding_config.embedding_model


@app.command("directory")
def load_directory(
    name: Annotated[str, typer.Option(help="Name of dataset to load.")],
    input_dir: Annotated[Optional[str], typer.Option(help="Path to directory containing dataset.")] = None,
    input_files: Annotated[List[str], typer.Option(help="List of paths to files containing dataset.")] = [],
    recursive: Annotated[bool, typer.Option(help="Recursively search for files in directory.")] = False,
    extensions: Annotated[str, typer.Option(help="Comma separated list of file extensions to load")] = default_extensions,
    user_id: Annotated[Optional[uuid.UUID], typer.Option(help="User ID to associate with dataset.")] = None,
    description: Annotated[Optional[str], typer.Option(help="Description of the source.")] = None,
):
    """
    Load a dataset from a directory or a list of files into the system.
    This function creates a data source, selects an embedding model, and attempts to load the data.
    In case of failure during data loading, it cleans up by deleting the created source.
    """
    client = create_client()

    # Create a DirectoryConnector instance based on the provided parameters
    connector = DirectoryConnector(
        input_files=input_files, input_directory=input_dir, recursive=recursive, extensions=extensions
    )

    # Get the list of available embedding configurations from the client
    embedding_configs = client.list_embedding_configs()

    # Select an appropriate embedding model
    embedding_model_name = select_embedding_model(embedding_configs)

    # Find the corresponding embedding configuration object
    embedding_config = next(
        (embedding_config for embedding_config in embedding_configs if embedding_config.embedding_model == embedding_model_name), None
    )
    assert embedding_config is not None, "Failed to find the selected embedding configuration"

    # Create a new data source with the chosen embedding configuration
    source = client.create_source(name=name, embedding_config=embedding_config)

    try:
        # Attempt to load the data using the connector and the newly created source
        client.load_data(connector, source_name=name)
    except Exception as e:
        # In case of failure, print an error message in red color and delete the source
        typer.secho(f"Failed to load data from provided information.\n{e}", fg=typer.colors.RED)
        client.delete_source(source.id)