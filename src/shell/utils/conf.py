import ast
import os
from enum import Enum
from typing import Annotated, List, Optional

import questionary
import typer
from prettytable.colortable import ColorTable, Themes
from tqdm import tqdm
from labo import utils

# Create a base client instance creator function to avoid repeated imports
def create_labo_client():
    from labo.client.client import create_client
    return create_client(base_url=os.getenv("MEMGPT_BASE_URL"), token=os.getenv("MEMGPT_SERVER_PASS"))

# Create a Typer application instance
app = typer.Typer()


@app.command()
def configure():
    """
    Inform the user that this command is deprecated and suggest an alternative command.
    """
    print("`labo configure` has been deprecated. Please see documentation on configuration, and run `labo run` instead.")


# Define an enumeration type to represent different list options
class ListChoice(str, Enum):
    agents = "agents"
    humans = "humans"
    personas = "personas"
    sources = "sources"


def _populate_agents_table(client, table):
    """
    Populate the table with agent-related information.
    """
    agents = client.list_agents()
    for agent in tqdm(agents):
        sources = client.list_attached_sources(agent_id=agent.id)
        source_names = [source.name for source in sources if source is not None]
        table.add_row(
            [
                agent.name,
                agent.llm_config.model,
                agent.embedding_config.embedding_model,
                agent.embedding_config.embedding_dim,
                agent.memory.get_block("persona").value[:100] + "...",
                agent.memory.get_block("human").value[:100] + "...",
                ",".join(source_names),
                utils.format_datetime(agent.created_at),
            ]
        )


def _populate_humans_table(client, table):
    """
    Populate the table with human-related information.
    """
    humans = client.list_humans()
    for human in humans:
        table.add_row([human.template_name, human.value.replace("\n", "")[:100]])


def _populate_personas_table(client, table):
    """
    Populate the table with persona-related information.
    """
    personas = client.list_personas()
    for persona in personas:
        table.add_row([persona.template_name, persona.value.replace("\n", "")[:100]])


def _populate_sources_table(client, table):
    """
    Populate the table with source-related information.
    """
    sources = client.list_sources()
    for source in sources:
        table.add_row(
            [
                source.name,
                source.description,
                source.embedding_config.embedding_model,
                source.embedding_config.embedding_dim,
                utils.format_datetime(source.created_at),
            ]
        )


@app.command()
def list(arg: Annotated[ListChoice, typer.Argument]):
    """
    Display a formatted table based on the selected option from the ListChoice enumeration.
    """
    client = create_labo_client()
    table = ColorTable(theme=Themes.OCEAN)
    table.field_names = {
        ListChoice.agents: ["Name", "LLM Model", "Embedding Model", "Embedding Dim", "Persona", "Human", "Data Source", "Create Time"],
        ListChoice.humans: ["Name", "Text"],
        ListChoice.personas: ["Name", "Text"],
        ListChoice.sources: ["Name", "Description", "Embedding Model", "Embedding Dim", "Created At"]
    }[arg]
    populate_table_func = {
        ListChoice.agents: _populate_agents_table,
        ListChoice.humans: _populate_humans_table,
        ListChoice.personas: _populate_personas_table,
        ListChoice.sources: _populate_sources_table
    }[arg]
    populate_table_func(client, table)
    print(table)


def extract_function_from_file(filename):
    """
    Extract the function definition from the given Python file.
    """
    with open(filename, "r", encoding="utf-8") as file:
        source_code = file.read()
    module = ast.parse(source_code)
    func_def = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if not func_def:
        raise ValueError("No function found in the provided file")
    return func_def


@app.command()
def add_tool(
    filename: str = typer.Option(..., help="Path to the Python file containing the function"),
    name: Optional[str] = typer.Option(None, help="Name of the tool"),
    update: bool = typer.Option(True, help="Update the tool if it already exists"),
    tags: Optional[List[str]] = typer.Option(None, help="Tags for the tool"),
):
    """
    Add or update a tool in the client using the function defined in the provided file.
    """
    client = create_labo_client()
    func_def = extract_function_from_file(filename)
    exec(compile(ast.Module([func_def], []), filename, "exec"))
    func = eval(func_def.name)
    tool = client.create_or_update_tool(func=func, name=name, tags=tags, update=update)
    print(f"Tool {tool.name} added successfully")


@app.command()
def list_tools():
    """
    Print the names of all tools available in the client.
    """
    client = create_labo_client()
    tools = client.list_tools()
    for tool in tools:
        print(f"Tool: {tool.name}")


def handle_entity_creation_or_update(client, option, name, text):
    """
    Handle the creation or update of an entity (human or persona) based on its existence.
    """
    if option == "persona":
        persona_id = client.get_persona_id(name)
        if persona_id:
            persona = client.get_persona(persona_id)
            if not questionary.confirm(f"Persona {name} already exists. Overwrite?").ask():
                return
            client.update_persona(persona_id, text=text)
        else:
            client.create_persona(name=name, text=text)
    elif option == "human":
        human_id = client.get_human_id(name)
        if human_id:
            human = client.get_human(human_id)
            if not questionary.confirm(f"Human {name} already exists. Overwrite?").ask():
                return
            client.update_human(human_id, text=text)
        else:
            client.create_human(name=name, text=text)
    else:
        raise ValueError(f"Unknown kind {option}")


@app.command()
def add(
    option: str,
    name: Annotated[str, typer.Option(help="Name of human/persona")],
    text: Annotated[Optional[str], typer.Option(help="Text of human/persona")] = None,
    filename: Annotated[Optional[str], typer.Option("-f", help="Specify filename")] = None,
):
    """
    Create or update a human or persona entity in the client based on provided parameters.
    """
    client = create_labo_client()
    if filename:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
    handle_entity_creation_or_update(client, option, name, text)


def handle_entity_deletion(client, option, name):
    """
    Handle the deletion of an entity (source, agent, human, or persona) based on its existence.
    """
    try:
        if option == "source":
            source_id = client.get_source_id(name)
            assert source_id is not None, f"Source {name} does not exist"
            client.delete_source(source_id)
        elif option == "agent":
            agent_id = client.get_agent_id(name)
            assert agent_id is not None, f"Agent {name} does not exist"
            client.delete_agent(agent_id=agent_id)
        elif option == "human":
            human_id = client.get_human_id(name)
            assert human_id is not None, f"Human {name} does not exist"
            client.delete_human(human_id)
        elif option == "persona":
            persona_id = client.get_persona_id(name)
            assert persona_id is not None, f"Persona {name} does not exist"
            client.delete_persona(persona_id)
        else:
            raise ValueError(f"Option {option} not implemented")
        print(f"Deleted {option} '{name}'")
    except Exception as e:
        print(f"Failed to delete {option}'{name}'\n{e}")


@app.command()
def delete(option: str, name: str):
    """
    Delete an entity from the client based on the provided option and name.
    """
    client = create_labo_client()
    handle_entity_deletion(client, option, name)