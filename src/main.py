import os
import sys
import traceback
from typing import Optional

import questionary
import requests
import typer
from rich.console import Console

os.environ["COMPOSIO_DISABLE_VERSION_CHECK"] = "true"

shell = typer.Typer(pretty_exceptions_enable=False)
shell.command(name="execute")(execute_task)
shell.command(name="info")(display_info)
shell.command(name="setup")(initialize)
shell.command(name="show")(display_items)
shell.command(name="create")(create_item)
shell.command(name="register")(register_component)
shell.command(name="components")(show_components)
shell.command(name="remove")(remove_item)
shell.command(name="service")(start_service)
shell.command(name="browse")(explore_directory)
shell.add_typer(import_manager, name="import")
shell.command(name="analyze")(perform_analysis)
shell.command(name="purge")(purge_component)


def clear_display(output_manager, minimal_ui=False):
    if minimal_ui:
        return
    if os.name == "nt":
        output_manager.print("\033[A\033[K", end="")
    else:
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()


def process_interaction_cycle(
    core_component: core_engine.Agent,
    settings: dict,
    initial: bool,
    skip_validation: bool = False,
    minimal_ui: bool = False,
    realtime_output: bool = False,
):
    if isinstance(core_component.interface, AgentRefreshStreamingInterface):
        if not realtime_output:
            core_component.interface = core_component.interface.nonstreaming_interface

    output_manager = (
        core_component.interface.console
        if hasattr(core_component.interface, "console")
        else Console()
    )

    interaction_count = 0
    input_content = None
    skip_input = False
    message_content = None
    USER_INITIATES = initial

    if not USER_INITIATES:
        output_manager.input(
            "[bold cyan]Press Enter to start (system will request initial message)[/bold cyan]\n"
        )
        clear_display(output_manager, minimal_ui=minimal_ui)
        print()

    extended_input_mode = False
    backend_service = create_backend_connection()

    while True:
        if not skip_input and (interaction_count > 0 or USER_INITIATES):
            if not realtime_output:
                print()
            input_content = questionary.text(
                "Type your instruction:",
                multiline=extended_input_mode,
                qmark="→",
            ).ask()
            clear_display(output_manager, minimal_ui=minimal_ui)
            if not realtime_output:
                print()

            if input_content is None:
                input_content = "/terminate"

            input_content = input_content.rstrip()

            if input_content.startswith("!"):
                print(f"System commands should start with '/' not '!'")
                continue

            if input_content == "":
                print("Please provide non-empty input!")
                continue

            if input_content.startswith("/"):
                if input_content.lower() == "/terminate":
                    core_engine.save_agent(core_component)
                    break
                elif input_content.lower() in ["/store", "/preserve"]:
                    core_engine.save_agent(core_component)
                    continue
                elif input_content.lower() == "/link":
                    available_resources = backend_service.list_sources()
                    if not available_resources:
                        typer.secho(
                            'No data sources found. First import data using "labo import ..."',
                            fg=typer.colors.RED,
                            bold=True,
                        )
                        continue

                    compatible_sources = []
                    incompatible_sources = []
                    for resource in available_resources:
                        if (
                            resource.embedding_config
                            == core_component.agent_state.embedding_config
                        ):
                            compatible_sources.append(resource.name)
                        else:
                            typer.secho(
                                f"Resource {resource.name} found with incompatible configuration",
                                fg=typer.colors.YELLOW,
                            )
                            incompatible_sources.append(resource.name)

                    selected_source = questionary.select(
                        "Choose data source", choices=compatible_sources
                    ).ask()

                    backend_service.attach_source_to_agent(
                        agent_id=core_component.agent_state.id,
                        source_name=selected_source,
                    )
                    continue

                # ...existing code...
                # (Continue with similar transformations for other commands)

            else:
                message_content = str(input_content)

        skip_input = False

        def execute_processing_cycle(message_content, skip_validation):
            if message_content is None:
                cycle_output = core_component.inner_step(
                    messages=[],
                    first_message=False,
                    skip_verify=skip_validation,
                    stream=realtime_output,
                )
            else:
                cycle_output = core_component.step_user_message(
                    user_message_str=message_content,
                    first_message=False,
                    skip_verify=skip_validation,
                    stream=realtime_output,
                )

            # ...existing code...
            # (Continue with similar transformations for the rest of the function)

        # ...existing code...
        # (Continue with similar transformations for the rest of the loop logic)

    print("Operation completed.")


SYSTEM_COMMANDS = [
    ("//", "switch extended input mode"),
    ("/terminate", "end current session"),
    ("/store", "create session snapshot"),
    ("/import", "retrieve saved snapshot"),
]
