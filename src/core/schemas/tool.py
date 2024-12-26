from typing import Dict, List, Optional

from pydantic import Field, model_validator

from labo.constants import FUNCTION_RETURN_CHAR_LIMIT
from labo.functions.functions import derive_openai_json_schema
from labo.functions.helpers import (
    generate_composio_tool_wrapper,
    generate_langchain_tool_wrapper,
)
from labo.functions.schema_generator import generate_schema_from_args_schema_v2
from labo.schemas.labo_base import LABOBase
from labo.schemas.openai.chat_completions import ToolCall


class BaseTool(LABOBase):
    """
    Represents the base structure for tools in the system.

    This class inherits from `LABOBase` and is likely used to establish a common foundation for more specific
    tool-related classes. It defines a prefix (`"tool"`) for the unique identifiers of tools, which helps in
    creating consistent and identifiable IDs for different tools within the system.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"tool"`, which is used to prefix the unique identifiers of
                       tools. This is likely used in the process of generating unique IDs for each tool
                       instance.
    """
    __id_prefix__ = "tool"


class Tool(BaseTool):
    """
    Represents a tool within the system, including its identifier, properties, and methods for validation and conversion.

    This class inherits from `BaseTool` and adds specific attributes that define a tool in more detail. It
    includes a unique ID generated using the `generate_id_field` method from the base class, along with
    various optional and required attributes related to the tool's characteristics, such as its description,
    source type, module, organization association, name, tags, source code, JSON schema, and response character
    limit. It also provides a validation method to populate missing fields based on available information and
    a method to convert the tool object into a dictionary format useful for certain operations related to
    tool calls.

    Attributes:
    - `id`: A unique identifier for the tool, generated using the mechanism provided by the base class. This ID
            is used to distinguish one tool from another within the system.
    - `description`: An optional string representing a description of the tool. This can provide more context
                     about what the tool does or how it should be used.
    - `source_type`: An optional string representing the type of the source code of the tool. For example, it
                     might indicate the programming language or a specific framework used.
    - `module`: An optional string representing the module of the function. This could refer to the Python
                module or a similar concept depending on the programming context.
    - `organization_id`: An optional string representing the unique identifier of the organization associated
                         with the tool. This helps in linking the tool to a particular organization within the
                         system.
    - `name`: An optional string representing the name of the function that the tool implements. If not provided,
              it can be populated from the JSON schema.
    - `tags`: A list of strings representing metadata tags for the tool. These can be used for categorization,
              searching, or other organizational purposes.
    - `source_code`: A required string representing the actual source code of the function that the tool
                     encapsulates.
    - `json_schema`: An optional dictionary representing the JSON schema of the function. This is useful for
                     defining the input and output structure of the tool's function and can be auto-generated
                     if not provided.
    - `return_char_limit`: An integer representing the maximum number of characters allowed in the response of
                           the tool's function. By default, it's set to the value from `FUNCTION_RETURN_CHAR_LIMIT`.
    - `created_by_id`: An optional string representing the ID of the user who created the tool. This can be
                       used to track the originator of the tool.
    - `last_updated_by_id`: An optional string representing the ID of the user who last updated the tool. This
                           helps in keeping track of who made the most recent modifications.

    Methods:
    - `populate_missing_fields`: A model validator method that runs after other validations. It checks if the
                                `json_schema`, `name`, or `description` fields are missing and populates them
                                using the `source_code` or existing `json_schema` information as appropriate.
                                For example, it can derive the JSON schema from the source code if it's not
                                already set, or use the name and description from the existing JSON schema if
                                they're missing. After populating the fields, it returns the updated tool object.
    - `to_dict`: A method that converts the tool object into a dictionary representation suitable for creating
                a `ToolCall` object. It constructs the dictionary with the tool's ID, a fixed tool call type
                ("function"), and the module as the function name.
    """
    id: str = BaseTool.generate_id_field()
    description: Optional[str] = Field(None, description="The description of the tool.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    module: Optional[str] = Field(None, description="The module of the function.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the tool.")
    name: Optional[str] = Field(None, description="The name of the function.")
    tags: List[str] = Field([], description="Metadata tags.")
    source_code: str = Field(..., description="The source code of the function.")
    json_schema: Optional[Dict] = Field(None, description="The JSON schema of the function.")
    return_char_limit: int = Field(FUNCTION_RETURN_CHAR_LIMIT, description="The maximum number of characters in the response.")
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this Tool.")

    @model_validator(mode="after")
    def populate_missing_fields(self):
        """
        Populate missing fields in the tool object using available information.

        This method checks if the `json_schema`, `name`, or `description` fields are not set and attempts to
        populate them based on other available data. If the `json_schema` is missing, it tries to derive it
        from the `source_code` using the `derive_openai_json_schema` function. If the `name` is not provided,
        it tries to get it from the `json_schema` dictionary. Similarly, if the `description` is missing,
        it attempts to retrieve it from the `json_schema` as well. After populating the relevant fields, the
        updated tool object is returned.

        Returns:
        - `Tool`: The updated tool object with populated missing fields.
        """
        if not self.json_schema:
            self.json_schema = derive_openai_json_schema(source_code=self.source_code)
        if not self.name:
            self.name = self.json_schema.get("name")
        if not self.description:
            self.description = self.json_schema.get("description")
        return self

    def to_dict(self):
        """
        Convert the tool object to a dictionary suitable for creating a `ToolCall` object.

        This method constructs a dictionary with the necessary information to create a `ToolCall` object.
        It includes the tool's unique ID, sets the tool call type to "function", and uses the module as the
        function name.

        Returns:
        - `dict`: A dictionary representation of the tool object in a format suitable for `ToolCall` object
                  creation.
        """
        return vars(
            ToolCall(
                tool_id=self.id,
                tool_call_type="function",
                function=self.module,
            )
        )


class ToolCreate(LABOBase):
    """
    Represents the data required or allowed for creating a new tool.

    This class inherits from `LABOBase` and defines the fields that can be provided or are relevant when
    creating a new tool. It allows for specifying details like the tool's name (which can be auto-populated
    from the source code if not given), description, tags, module, source code, source type, and JSON schema
    (which can also be auto-generated if missing). It also provides class methods to create tool instances
    based on external tool sets from `Composio` and `LangChain` and methods to load default sets of tools
    from these frameworks.

    Attributes:
    - `name`: An optional string representing the name of the function. If not provided, it can be auto-generated
              from the `source_code`.
    - `description`: An optional string representing the description of the tool.
    - `tags`: A list of strings representing metadata tags for the tool. By default, it's an empty list.
    - `module`: An optional string representing the source code of the function.
    - `source_code`: A required string representing the actual source code of the function that the tool will
                     encapsulate.
    - `source_type`: A string representing the source type of the function. By default, it's set to "python".
    - `json_schema`: An optional dictionary representing the JSON schema of the function. If not provided, it
                     can be auto-generated from the `source_code`.
    - `return_char_limit`: An integer representing the maximum number of characters allowed in the response of
                           the tool's function. By default, it's set to the value from `FUNCTION_RETURN_CHAR_LIMIT`.

    Class Methods:
    - `from_composio`: A class method that creates a `ToolCreate` instance from a `Composio` tool. It takes an
                       `action_name` and an optional `api_key` as parameters. It initializes a `ComposioToolSet`,
                       retrieves the relevant `Composio` tools based on the `action_name`, validates that only
                       one matching tool is found, and then uses the details of that tool (description, source
                       type, tags, and generates a wrapper function and JSON schema) to create and return a
                       `ToolCreate` instance.
    - `from_langchain`: A class method that creates a `ToolCreate` instance from a `LangChain` tool. It takes a
                        `LangChainBaseTool` object and an optional `additional_imports_module_attr_map` dictionary
                        as parameters. It extracts the relevant details from the `LangChain` tool (description,
                        source type, tags), generates a wrapper function and JSON schema based on the tool's
                        `args_schema`, and returns a `ToolCreate` instance with these populated fields.
    - `load_default_langchain_tools`: A class method that loads the default `LangChain` tools. Currently, it
                                     creates a single `ToolCreate` instance from the `WikipediaQueryRun` tool
                                     using the `WikipediaAPIWrapper` and returns a list containing that single
                                     tool instance.
    - `load_default_composio_tools`: A class method that is intended to load the default `Composio` tools.
                                     Currently, it simply returns an empty list, but it might be implemented
                                     in the future to return a list of `ToolCreate` instances representing the
                                     default `Composio` tools.
    """
    name: Optional[str] = Field(None, description="The name of the function (auto-generated from source_code if not provided).")
    description: Optional[str] = Field(None, description="The description of the tool.")
    tags: List[str] = Field([], description="Metadata tags.")
    module: Optional[str] = Field(None, description="The source code of the function.")
    source_code: str = Field(..., description="The source code of the function.")
    source_type: str = Field("python", description="The source type of the function.")
    json_schema: Optional[Dict] = Field(
        None, description="The JSON schema of the function (auto-generated from source_code if not provided)"
    )
    return_char_limit: int = Field(FUNCTION_RETURN_CHAR_LIMIT, description="The maximum number of characters in the response.")

    @classmethod
    def from_composio(cls, action_name: str, api_key: Optional[str] = None) -> "ToolCreate":
        """
        Create a `ToolCreate` instance from a `Composio` tool.

        This class method takes an `action_name` (which likely identifies a specific action or tool within
        the `Composio` framework) and an optional `api_key` as parameters. It initializes a `ComposioToolSet`
        with the appropriate logging level and, if provided, the `api_key`. It then retrieves the `Composio`
        tools that match the given `action_name`, validates that exactly one matching tool is found, and uses
        the details of that tool (such as its description, source type, tags) to generate a wrapper function
        and a JSON schema. Finally, it creates and returns a `ToolCreate` instance populated with these
        details.

        Args:
        - `action_name`: A string representing the name of the action or tool within the `Composio` framework.
        - `api_key`: An optional string representing the API key for accessing the `Composio` tools. If not
                     provided, a default initialization without the key is used.

        Returns:
        - `ToolCreate`: A `ToolCreate` instance representing the `Composio` tool with its relevant details
                        populated.
        """
        from composio import LogLevel
        from composio_langchain import ComposioToolSet

        if api_key:
            composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR, api_key=api_key)
        else:
            composio_toolset = ComposioToolSet(logging_level=LogLevel.ERROR)
        composio_tools = composio_toolset.get_tools(actions=[action_name])
        assert len(composio_tools) > 0, "User supplied parameters do not match any Composio tools"
        assert len(composio_tools) == 1, f"User supplied parameters match too many Composio tools; {len(composio_tools)} > 1"
        composio_tool = composio_tools[0]

        description = composio_tool.description
        source_type = "python"
        tags = ["composio"]
        wrapper_func_name, wrapper_function_str = generate_composio_tool_wrapper(action_name)
        json_schema = generate_schema_from_args_schema_v2(composio_tool.args_schema, name=wrapper_func_name, description=description)

        return cls(
            name=wrapper_func_name,
            description=description,
            source_type=source_type,
            tags=tags,
            source_code=wrapper_function_str,
            json_schema=json_schema,
        )

    @classmethod
    def from_langchain(
        cls,
        langchain_tool: "LangChainBaseTool",
        additional_imports_module_attr_map: dict[str, str] = None,
    ) -> "ToolCreate":
        """
        Create a `ToolCreate` instance from a `LangChain` tool.

        This class method takes a `LangChainBaseTool` object and an optional `additional_imports_module_attr_map`
        dictionary as parameters. It extracts the description of the `LangChain` tool, sets the source type
        to "python", and creates metadata tags indicating it's related to `LangChain`. It then generates a
        wrapper function and a JSON schema based on the tool's `args_schema`, name, and description. Finally,
        it creates and returns a `ToolCreate` instance populated with these details.

        Args:
        - `langchain_tool`: A `LangChainBaseTool` object representing the `LangChain` tool from which the
                            `ToolCreate` instance will be created.
        - `additional_imports_module_attr_map`: An optional dictionary mapping additional import module names
                                                to their corresponding attribute names. This can be used for
                                                handling imports and attributes in the context of generating
                                                the wrapper function.

        Returns:
        - `ToolCreate`: A `ToolCreate` instance representing the `LangChain` tool with its relevant details
                        populated.
        """
        description = langchain_tool.description
        source_type = "python"
        tags = ["langchain"]