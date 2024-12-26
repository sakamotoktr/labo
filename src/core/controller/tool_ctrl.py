import importlib
import inspect
import warnings
from typing import List, Optional


class ToolHandler:
    """Class to manage operations and logic related to tools."""

    CORE_TOOL_NAMES = [
        "send_message",
        "conversation_lookup",
        "memory_insert",
        "memory_lookup",
    ]
    CORE_MEMORY_TOOL_NAMES = ["core_memory_add", "core_memory_update"]

    def __init__(self):
        # Initializing the session maker similar to OrganizationManager
        self.session_provider = db_session_provider

    @enforce_type_safety
    def upsert_or_modify_tool(
        self, tool_data: ToolSchema, user: UserSchema
    ) -> ToolSchema:
        """Insert or modify a tool based on input data."""
        # Retrieve the existing tool
        existing_tool = self.get_tool_by_name(tool_name=tool_data.name, user=user)
        if existing_tool:
            # Prepare the data for updating
            update_fields = tool_data.model_dump(
                exclude={"module"}, exclude_unset=True, exclude_none=True
            )

            if update_fields:
                self.modify_tool_by_id(
                    existing_tool.id, ToolModification(**update_fields), user
                )
            else:
                debug_print(
                    f"Attempted to upsert tool for user {user.id}, organization {user.organization_id}, tool {tool_data.name}, but no updates found."
                )
        else:
            # Create a new tool if not found
            existing_tool = self.create_new_tool(tool_data, user)

        return existing_tool

    @enforce_type_safety
    def create_new_tool(self, tool_data: ToolSchema, user: UserSchema) -> ToolSchema:
        """Create and save a new tool to the database."""
        with self.session_provider() as session:
            tool_data.organization_id = user.organization_id
            if tool_data.description is None:
                tool_data.description = tool_data.json_schema.get("description", None)
            tool_dict = tool_data.model_dump()
            new_tool = ToolDBModel(**tool_dict)
            new_tool.create(session, user=user)
        return new_tool.to_pydantic()

    @enforce_type_safety
    def get_tool_by_id(self, tool_id: str, user: UserSchema) -> ToolSchema:
        """Fetch a tool using its unique ID."""
        with self.session_provider() as session:
            tool = ToolDBModel.read(session, identifier=tool_id, user=user)
            return tool.to_pydantic()

    @enforce_type_safety
    def get_tool_by_name(
        self, tool_name: str, user: UserSchema
    ) -> Optional[ToolSchema]:
        """Fetch a tool by its name for a specific user."""
        try:
            with self.session_provider() as session:
                tool = ToolDBModel.read(session, name=tool_name, user=user)
                return tool.to_pydantic()
        except NoResultFound:
            return None

    @enforce_type_safety
    def list_all_tools(
        self, user: UserSchema, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[ToolSchema]:
        """Fetch a list of tools, with optional pagination."""
        with self.session_provider() as session:
            tools = ToolDBModel.list(
                session,
                cursor=cursor,
                limit=limit,
                organization_id=user.organization_id,
            )
            return [tool.to_pydantic() for tool in tools]

    @enforce_type_safety
    def modify_tool_by_id(
        self, tool_id: str, tool_update: ToolModification, user: UserSchema
    ) -> ToolSchema:
        """Update an existing tool using its ID."""
        with self.session_provider() as session:
            tool = ToolDBModel.read(session, identifier=tool_id, user=user)

            # Apply the updates
            update_fields = tool_update.model_dump(exclude_none=True)
            for field, value in update_fields.items():
                setattr(tool, field, value)

            # Auto-refresh schema if source code changes but no new schema provided
            if "source_code" in update_fields and "json_schema" not in update_fields:
                tool_schema = tool.to_pydantic()
                new_schema = fetch_openai_schema(source_code=tool_schema.source_code)
                tool.json_schema = new_schema

            return tool.update(session, user=user).to_pydantic()

    @enforce_type_safety
    def remove_tool_by_id(self, tool_id: str, user: UserSchema) -> None:
        """Delete a tool based on its ID."""
        with self.session_provider() as session:
            try:
                tool = ToolDBModel.read(session, identifier=tool_id, user=user)
                tool.hard_delete(session, user=user)
            except NoResultFound:
                raise ValueError(f"Tool with ID {tool_id} does not exist.")

    @enforce_type_safety
    def add_default_tools(self, user: UserSchema) -> List[ToolSchema]:
        """Add pre-defined default tools."""
        module_name = "core"
        module_full_name = f"labo.functions.function_sets.{module_name}"
        try:
            module = importlib.import_module(module_full_name)
        except Exception as e:
            raise e

        function_schemas = []
        try:
            function_schemas = load_function_definitions(module)
        except ValueError as e:
            warnings.warn(f"Failed to load function set '{module_name}': {e}")

        tools = []
        for tool_name, schema in function_schemas.items():
            if tool_name in CORE_FUNCTIONS + CORE_MEMORY_TOOLS:
                source_code = inspect.getsource(schema["python_function"])
                tags = [module_name]
                if module_name == "core":
                    tags.append("labo-core")

                tools.append(
                    self.upsert_or_modify_tool(
                        ToolSchema(
                            name=tool_name,
                            tags=tags,
                            source_type="python",
                            module=schema["module"],
                            source_code=source_code,
                            json_schema=schema["json_schema"],
                        ),
                        user=user,
                    )
                )

        return tools
