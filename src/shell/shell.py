import logging
import sys
from enum import Enum
from typing import Annotated, Optional

import questionary
import typer

import labo.utils as utils
from labo import create_client
from labo.agent import Agent, save_agent
from labo.config import LABOConfig
from labo.constants import (
    CLI_WARNING_PREFIX,
    CORE_MEMORY_BLOCK_CHAR_LIMIT,
    LABO_DIR,
    MIN_CONTEXT_WINDOW,
)
from labo.local_llm.constants import ASSISTANT_MESSAGE_CLI_SYMBOL
from labo.utils.log import get_logger
from labo.schemas.enums import OptionState
from labo.schemas.memory import ChatMemory, Memory
from labo.server.server import logger as server_logger
from labo.streaming_interface import (
    StreamingRefreshCLIInterface as interface,
)

logger = get_logger(__name__)


def open_folder():
    """
    Attempt to open the specified folder in the system explorer.
    If an error occurs during the process, print the error message.
    """
    try:
        print(f"Opening home folder: {LABO_DIR}")
        utils.open_folder_in_explorer(LABO_DIR)
    except Exception as e:
        print(f"Failed to open folder with system viewer, error:\n{e}")


class ServerChoice(Enum):
    rest_api = "rest"
    ws_api = "websocket"


def start_rest_server(
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    debug: Annotated[bool, typer.Option(help="Turn debugging output on")] = False,
):
    """
    Start the REST API server.
    If a KeyboardInterrupt is received (e.g., user presses Ctrl+C), gracefully terminate the server.
    """
    try:
        from labo.server.rest_api.app import start_server

        start_server(port=port, host=host, debug=debug)
    except KeyboardInterrupt:
        typer.secho("Terminating the server...")
        sys.exit(0)


def server(
    type: Annotated[ServerChoice, typer.Option(help="Server to run")] = "rest",
    port: Annotated[Optional[int], typer.Option(help="Port to run the server on")] = None,
    host: Annotated[Optional[str], typer.Option(help="Host to run the server on (default to localhost)")] = None,
    debug: Annotated[bool, typer.Option(help="Turn debugging output on")] = False,
    ade: Annotated[bool, typer.Option(help="Allows remote access")] = False,
    secure: Annotated[bool, typer.Option(help="Adds simple security access")] = False,
    localhttps: Annotated[bool, typer.Option(help="Setup local https")] = False,
):
    """
    Start the specified server (either REST API or WebSocket API).
    Currently, the WebSocket API support is marked as deprecated and raises a NotImplementedError.
    """
    if type == ServerChoice.rest_api:
        start_rest_server(port=port, host=host, debug=debug)
    elif type == ServerChoice.ws_api:
        raise NotImplementedError("WS suppport deprecated")


def select_agent_if_needed(client, yes, agents):
    """
    Prompt the user to select an existing agent if needed.
    If the user chooses to select an agent, present a list of available agents and return the selected one.
    If the user cancels the operation (by pressing Ctrl+C or closing the prompt), raise a KeyboardInterrupt.
    """
    if not yes and len(agents) > 0:
        print()
        select_agent = questionary.confirm("Would you like to select an existing agent?").ask()
        if select_agent is None:
            raise KeyboardInterrupt
        if select_agent:
            agents = [a.name for a in agents]
            return questionary.select("Select agent:", choices=agents).ask()
    return None


def select_llm_model(client):
    """
    Prompt the user to select an LLM model from the available options.
    If there are no available models, raise a ValueError.
    If there's only one option, return that model.
    Otherwise, present a list of models and return the user's selected one.
    """
    llm_configs = client.list_llm_configs()
    llm_options = [llm_config.model for llm_config in llm_configs]
    if len(llm_options) == 0:
        raise ValueError("No LLM models found. Please enable a provider.")
    elif len(llm_options) == 1:
        return llm_options[0]
    else:
        llm_choices = [questionary.Choice(title=llm_config.pretty_print(), value=llm_config) for llm_config in llm_configs]
        return questionary.select("Select LLM model:", choices=llm_choices).ask().model


def select_embedding_model(client):
    """
    Prompt the user to select an embedding model from the available options.
    If there are no available models, raise a ValueError.
    If there's only one option, return that model.
    Otherwise, present a list of models and return the user's selected one.
    """
    embedding_configs = client.list_embedding_configs()
    embedding_options = [embedding_config.embedding_model for embedding_config in embedding_configs]
    if len(embedding_options) == 0:
        raise ValueError("No embedding models found. Please enable a provider.")
    elif len(embedding_options) == 1:
        return embedding_options[0]
    else:
        embedding_choices = [
            questionary.Choice(title=embedding_config.pretty_print(), value=embedding_config) for embedding_config in embedding_configs
        ]
        return questionary.select("Select embedding model:", choices=embedding_choices).ask().embedding_model


def load_system_prompt(system, system_file):
    """
    Load the system prompt either from a provided file or use the provided text.
    If the system file is specified but not found, print an error message and exit.
    """
    if system_file:
        try:
            with open(system_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            typer.secho(f"System file not found at {system_file}", fg=typer.colors.RED)
            sys.exit(1)
    return system


def run(
    persona: Annotated[Optional[str], typer.Option(help="Specify persona")] = None,
    agent: Annotated[Optional[str], typer.Option(help="Specify agent name")] = None,
    human: Annotated[Optional[str], typer.Option(help="Specify human")] = None,
    system: Annotated[Optional[str], typer.Option(help="Specify system prompt (raw text)")] = None,
    system_file: Annotated[Optional[str], typer.Option(help="Specify raw text file containing system prompt")] = None,
    model: Annotated[Optional[str], typer.Option(help="Specify the LLM model")] = None,
    model_wrapper: Annotated[Optional[str], typer.Option(help="Specify the LLM model wrapper")] = None,
    model_endpoint: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint")] = None,
    model_endpoint_type: Annotated[Optional[str], typer.Option(help="Specify the LLM model endpoint type")] = None,
    context_window: Annotated[
        Optional[int], typer.Option(help="The context window of the LLM you are using (e.g. 8k for most Mistral 7B variants)")
    ] = None,
    core_memory_limit: Annotated[
        Optional[int], typer.Option(help="The character limit to each core-memory section (human/persona).")
    ] = CORE_MEMORY_BLOCK_CHAR_LIMIT,
    first: Annotated[bool, typer.Option(help="Use --first to send the first message in the sequence")] = False,
    strip_ui: Annotated[bool, typer.Option(help="Remove all the bells and whistles in CLI output (helpful for testing)")] = False,
    debug: Annotated[bool, typer.Option(help="Use --debug to enable debugging output")] = False,
    no_verify: Annotated[bool, typer.Option(help="Bypass message verification")] = False,
    yes: Annotated[bool, typer.Option("-y", help="Skip confirmation prompt and use defaults")] = False,
    stream: Annotated[bool, typer.Option(help="Enables message streaming in the CLI (if the backend supports it)")] = False,
    no_content: Annotated[
        OptionState, typer.Option(help="Set to 'yes' for LLM APIs that omit the `content` field during tool calling")
    ] = OptionState.DEFAULT,
):
    """
    Main function to run an agent.
    It configures logging based on the debug flag, loads or creates an agent,
    and then runs the agent loop with the specified parameters.
    """
    utils.DEBUG = debug
    if debug:
        logger.setLevel(logging.DEBUG)
        server_logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)
        server_logger.setLevel(logging.CRITICAL)

    config = LABOConfig.load()
    client = create_client()

    # Check if user wants to select an existing agent
    agent_name = select_agent_if_needed(client, yes, client.list_agents())
    if agent_name:
        agent_id = client.get_agent_id(agent_name)
        agent_state = client.get_agent(agent_id)
    else:
        agent_state = None

    human = human if human else config.human
    persona = persona if persona else config.persona

    if agent_state:
        typer.secho(f"\n🔁 Using existing agent {agent_name}", fg=typer.colors.GREEN)
        printd("Loading agent state:", agent_state.id)
        printd("Agent state:", agent_state.name)

        # Update agent configuration if new values are provided
        if model and model!= agent_state.llm_config.model:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model {agent_state.llm_config.model} with {model}", fg=typer.colors.YELLOW
            )
            agent_state.llm_config.model = model
        if context_window is not None and int(context_window)!= agent_state.llm_config.context_window:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing context window {agent_state.llm_config.context_window} with {context_window}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.context_window = context_window
        if model_wrapper and model_wrapper!= agent_state.llm_config.model_wrapper:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model wrapper {agent_state.llm_config.model_wrapper} with {model_wrapper}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_wrapper = model_wrapper
        if model_endpoint and model_endpoint!= agent_state.llm_config.model_endpoint:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint {agent_state.llm_config.model_endpoint} with {model_endpoint}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint = model_endpoint
        if model_endpoint_type and model_endpoint_type!= agent_state.llm_config.model_endpoint_type:
            typer.secho(
                f"{CLI_WARNING_PREFIX}Overriding existing model endpoint type {agent_state.llm_config.model_endpoint_type} with {model_endpoint_type}",
                fg=typer.colors.YELLOW,
            )
            agent_state.llm_config.model_endpoint_type = model_endpoint_type

        agent_state = client.update_agent(
            agent_id=agent_state.id,
            name=agent_state.name,
            llm_config=agent_state.llm_config,
            embedding_config=agent_state.embedding_config,
        )

        labo_agent = Agent(agent_state=agent_state, interface=interface(), user=client.user)
    else:
        typer.secho("\n🧬 Creating new agent...", fg=typer.colors.WHITE)
        agent_name = agent if agent else utils.create_random_username()

        # Select LLM model
        llm_model_name = select_llm_model(client)
        llm_config = [llm_config for llm_config in client.list_llm_configs() if llm_config.model == llm_model_name][0]

        # Set context window if needed
        if llm_config.context_window is not None:
            context_window_validator = lambda x: x.isdigit() and int(x) > MIN_CONTEXT_WINDOW and int(x) <= llm_config.context_window
            context_window_input = questionary.text(
                "Select LLM context window limit (hit enter for default):",
                default=str(llm_config.context_window),
                validate=context_window_validator,
            ).ask()
            if context_window_input is not None:
                llm_config.context_window = int(context_window_input)
            else:
                sys.exit(1)

        # Select embedding model
        embedding_model_name = select_embedding_model(client)
        embedding_config = [
            embedding_config for embedding_config in client.list_embedding_configs() if embedding_config.embedding_model == embedding_model_name
        ][0]

        human_obj = client.get_human(client.get_human_id(name=human))
        if human_obj is None:
            typer.secho(f"Couldn't find human {human} in database, please run `labo add human`", fg=typer.colors.RED)
            sys.exit(1)
        persona_obj = client.get_persona(client.get_persona_id(name=persona))
        if persona_obj is None:
            typer.secho(f"Couldn't find persona {persona} in database, please run `labo add persona`", fg=typer.colors.RED)
            sys.exit(1)

        system_prompt = load_system_prompt(system, system_file)

        memory = ChatMemory(human=human_obj.value, persona=persona_obj.value, limit=core_memory_limit)
        metadata = {"human": human_obj.template_name, "persona": persona_obj.template_name}

        typer.secho(f"->  {ASSISTANT_MESSAGE_CLI_SYMBOL} Using persona profile: '{persona_obj.template_name}'", fg=typer.colors.WHITE)
        typer.secho(f"->  🧑 Using human profile: '{human_obj.template_name}'", fg=typer.colors.WHITE)

        agent_state = client.create_agent(
            name=agent_name,
            system=system_prompt,
            embedding_config=embedding_config,
            llm_config=llm_config,
            memory=memory,
            metadata=metadata,
        )
        assert isinstance(agent_state.memory, Memory), f"Expected Memory, got {type(agent_state.memory)}"
        typer.secho(f"->  🛠️  {len(agent_state.tools)} tools: {', '.join([t.name for t in agent_state.tools])}", fg=typer.colors.WHITE)

        labo_agent = Agent(
            interface=interface(),
            agent_state=client.get_agent(agent_state.id),
            first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
            user=client.user,
        )
        save_agent(labo_agent)
        typer.secho(f"🎉 Created new agent '{labo_agent.agent_state.name}' (id={labo_agent.agent_state.id})", fg=typer.colors.GREEN)

    from labo.main import run_agent_loop

    print()
    run_agent_loop(
        labo_agent=labo_agent,
        config=config,
        first=first,
        no_verify=no_verify,
        stream=stream,
    )


def delete_agent(
    agent_name: Annotated[str, typer.Option(help="Specify agent to delete")],
):
    """
    Delete a specified agent.
    Prompt the user for confirmation before deleting.
    If the agent is not found or deletion fails, print an appropriate error message.
    """
    config = LABOConfig.load()
    MetadataStore(config)
    client = create_client()
    agent = client.get_agent_by_name(agent_name)
    if not agent:
        typer.secho(f"Couldn't find agent named '{agent_name}' to delete", fg=typer.colors.RED)
        sys.exit(1)

    confirm = questionary.confirm(f"Are you sure you want to delete agent '{agent_name}' (id={agent.id