import inspect
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from docstring_parser import parse
from pydantic import BaseModel
from pydantic.v1 import BaseModel as V1BaseModel


def is_optional_type(annotation):
    """
    Check if the given annotation represents an optional type (i.e., a Union type that includes the None type).

    :param annotation: The type annotation to check.
    :return: True if it's an optional type, False otherwise.
    """
    if getattr(annotation, "__origin__", None) is Union:
        return type(None) in annotation.__args__
    return False


def get_optional_type_length(annotation):
    """
    Get the number of actual type arguments in an optional type (excluding the None type).
    Raises a ValueError if the given annotation is not an optional type.

    :param annotation: The type annotation to check.
    :return: The number of actual type arguments in the optional type (should be 1).
    """
    if is_optional_type(annotation):
        return len(annotation.__args__) - 1
    else:
        raise ValueError("The annotation is not an Optional type")


def python_type_to_json_schema_type(py_type) -> dict:
    """
    Convert a Python type to its corresponding JSON Schema type representation.

    :param py_type: The Python type to convert.
    :return: A dictionary representing the JSON Schema type.
    """
    if is_optional_type(py_type):
        type_args = get_args(py_type)
        assert get_optional_type_length(py_type) == 1, f"Optional type must have exactly one type argument, but got {py_type}"
        return python_type_to_json_schema_type(type_args[0])

    if get_origin(py_type) is Union:
        raise NotImplementedError("General Union types are not yet supported")

    origin = get_origin(py_type)
    if py_type == list or origin in (list, List):
        args = get_args(py_type)
        if args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
            return {
                "type": "array",
                "items": pydantic_model_to_json_schema(args[0]),
            }
        return {
            "type": "array",
            "items": python_type_to_json_schema_type(args[0]),
        }

    if py_type == dict or origin in (dict, Dict):
        args = get_args(py_type)
        if not args:
            return {
                "type": "object",
            }
        else:
            raise ValueError(
                f"Dictionary types {py_type} with nested type arguments are not supported (consider using a Pydantic model instead)"
            )

    if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        return pydantic_model_to_json_schema(py_type)

    type_mapping = {
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
        None: "null",
    }
    if py_type not in type_mapping:
        raise ValueError(f"Python type {py_type} has no corresponding JSON schema type - full map: {type_mapping}")
    else:
        return {"type": type_mapping[py_type]}


def pydantic_model_to_open_ai_schema(model: Type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to an OpenAI-compatible schema representation.

    :param model: The Pydantic model class.
    :return: A dictionary representing the OpenAI-compatible schema.
    """
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
    for param in docstring.params:
        if (param_name := param.arg_name) in parameters["properties"] and (param_description := param.description):
            if "description" not in parameters["properties"][param_name]:
                parameters["properties"][param_name]["description"] = param_description

    parameters["required"] = sorted(k for k, v in parameters["properties"].items() if "default" not in v)

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            raise ValueError(f"No description found in docstring or description field (model: {model}, docstring: {docstring})")

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


def pydantic_model_to_json_schema(model: Type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a JSON Schema representation and perform some cleaning and normalization operations.

    :param model: The Pydantic model class.
    :return: A dictionary representing the cleaned and normalized JSON Schema.
    """
    schema = model.model_json_schema()

    def clean_property(prop: dict) -> dict:
        """
        Clean a property dictionary to ensure it has a 'description' key and handle the 'type' appropriately.

        :param prop: The property dictionary to clean.
        :return: The cleaned property dictionary.
        """
        if "description" not in prop:
            raise ValueError(f"Property {prop} lacks a 'description' key")

        return {
            "type": "string" if prop["type"] == "string" else prop["type"],
            "description": prop["description"],
        }

    def resolve_reference(ref: str, schema: dict) -> dict:
        """
        Resolve a reference ($ref) in a JSON Schema to its actual schema definition.

        :param ref: The reference string.
        :param schema: The full JSON Schema dictionary containing all schema definitions.
        :return: The resolved actual schema definition dictionary.
        """
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unexpected reference format: {ref}")

        model_name = ref.split("/")[-1]
        if model_name not in schema.get("$defs", {}):
            raise ValueError(f"Reference {model_name} not found in schema definitions")

        return schema["$defs"][model_name]

    def clean_schema_part(schema_part: dict, full_schema: dict) -> dict:
        """
        Clean and normalize parts of a JSON Schema, handling references, arrays, objects, etc.

        :param schema_part: The part of the JSON Schema dictionary to clean and normalize.
        :param full_schema: The full JSON Schema dictionary containing all schema definitions.
        :return: The cleaned and normalized part of the JSON Schema dictionary.
        """
        if "$ref" in schema_part:
            schema_part = resolve_reference(schema_part["$ref"], full_schema)

        if "type" not in schema_part:
            raise ValueError(f"Schema part lacks a 'type' key: {schema_part}")

        if schema_part["type"] == "array":
            items_schema = schema_part["items"]
            if "$ref" in items_schema:
                items_schema = resolve_reference(items_schema["$ref"], full_schema)
            return {
                "type": "array",
                "items": clean_schema_part(items_schema, full_schema),
                "description": schema_part.get("description", ""),
            }

        if schema_part["type"] == "object":
            if "properties" not in schema_part:
                raise ValueError(f"Object schema lacks 'properties' key: {schema_part}")

            properties = {}
            for prop_name, prop in schema_part["properties"].items():
                if "items" in prop:
                    if "description" not in prop:
                        raise ValueError(f"Property {prop} lacks a 'description' key")
                    properties[prop_name] = {
                        "type": "array",
                        "items": clean_schema_part(prop["items"], full_schema),
                        "description": prop["description"],
                    }
                else:
                    properties[prop_name] = clean_property(prop)

            json_model_schema_dict = {
                "type": "object",
                "properties": properties,
                "required": schema_part.get("required", []),
            }
            if "description" in schema_part:
                json_model_schema_dict["description"] = schema_part["description"]

            return json_model_schema_dict

        return clean_property(schema_part)

    return clean_schema_part(schema_part=schema, full_schema=schema)


def generate_function_schema(
    function, function_name: Optional[str] = None, function_description: Optional[str] = None
) -> dict:
    """
    Generate a JSON Schema representation for the function call of the given function, including information
    about parameter types, descriptions, and whether they are required.

    :param function: The function for which to generate the schema.
    :param function_name: An optional function name. If not provided, the function's own name will be used.
    :param function_description: An optional function description. If not provided, the short description from
                                  the function's docstring will be used.
    :return: A dictionary representing the JSON Schema for the function call.
    """
    sig = inspect.signature(function)
    docstring = parse(function.__doc__)

    schema = {
        "name": function.__name__ if function_name is None else function_name,
        "description": docstring.short_description if function_description is None else function_description,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    for param in sig.parameters.values():
        if param.name in ["self", "agent_state"]:
            continue

        if param.annotation == inspect.Parameter.empty:
            raise TypeError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a type annotation")

        param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)

        if not param_doc or not param_doc.description:
            raise ValueError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a description in the docstring")

        if (
            (inspect.isclass(param.annotation) or inspect.isclass(get_origin(param.annotation) or param.annotation))
            and not get_origin(param.annotation)
            and issubclass(param.annotation, BaseModel)
        ):
            schema["parameters"]["properties"][param.name] = pydantic_model_to_json_schema(param.annotation)
            schema["parameters"]["properties"][param.name]["description"] = param_doc.description
        else:
            param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
            if not param_doc:
                raise ValueError(f"Parameter '{param.name}' in function '{function.__name__}' lacks a description in the docstring")
            elif not isinstance(param_doc.description, str):
                raise ValueError(
                    f"Parameter '{param.name}' in function '{function.__name__}' has a description in the docstring that is not a string (type: {type(param_doc.description)})"
                )
            else:
                if param.annotation!= inspect.Parameter.empty:
                    param_generated_schema = python_type_to_json_schema_type(param.annotation)
                else:
                    param_generated_schema = {"type": "string"}

                param_generated_schema["description"] = param_doc.description

                schema["parameters"]["properties"][param.name] = param_generated_schema

        if param.default == inspect.Parameter.empty and not is_optional_type(param.annotation):
            schema["parameters"]["required"].append(param.name)

        if get_origin(param.annotation) is list:
            if get_args(param.annotation)[0] is str:
                schema["parameters"]["properties"][param.name]["items"] = {"type": "string"}

        if param.annotation == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

    if function.__name__ not in ["send_message"]:
        schema["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        schema["parameters"]["required"].append("request_heartbeat")

    return schema


def generate_schema_from_args_schema_version_1(
    args_schema: Type[V1BaseModel], name: Optional[str] = None, description: Optional[str] = None, append_heartbeat: bool = True
) -> Dict[str, Any]:
    """
    Generate a JSON Schema representation for the function call based on the Pydantic V1 version of the args schema.
    Optionally, a heartbeat request field can be appended.

    :param args_schema: The Pydantic V1 version of the args schema class.
    :param name: An optional function name.
    :param description: An optional function description.
    :param append_heartbeat: Whether to append the heartbeat request field. Defaults to True.
    :return: A dictionary representing the JSON Schema for the function call.
    """
    properties = {}
    required = []
    for field_name, field in args_schema.__fields__.items():
        if field.type_ == str:
            field_type = "string"
        elif field.type_ == int:
            field_type = "integer"
        elif field.type_ == bool:
            field_type = "boolean"
        else:
            field_type = field.type_.__name__

        properties[field_name] = {
            "type": field_type,
            "description": field.field_info.description,
        }
        if field.required:
            required.append(field_name)

    function_call_json = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    if append_heartbeat:
        function_call_json["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        function_call_json["parameters"]["required"].append("request_heartbeat")

    return function_call_json


def generate_schema_from_args_schema_version_2(
    args_schema: Type[BaseModel], name: Optional[str] = None, description: Optional[str] = None, append_heartbeat: bool = True
) -> Dict[str, Any]:
    """
    Generate a JSON Schema representation for the function call based on the Pydantic version 2 of the args schema.
    Optionally, a heartbeat request field can be appended.

    :param args_schema: The Pydantic version 2 of the args schema class.
    :param name: An optional function name.
    :param description: An optional function description.
    :param append_heartbeat: Whether to append the heartbeat request field. Defaults to True.
    :return: A dictionary representing the JSON Schema for the function call.
    """
    properties = {}
    required = []
    for field_name, field in args_schema.model_fields.items():
        field_type_annotation = field.annotation
        if field_type_annotation == str:
            field_type = "string"
        elif field_type_annotation == int:
            field_type = "integer"
        elif field_type_annotation == bool:
            field_type = "boolean"
        else:
            field_type = field_type_annotation.__name__

        properties[field_name] = {
            "type": field_type,
            "description": field.description,
        }
        if field.is_required():
            required.append(field_name)

    function_call_json = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }

    if append_heartbeat:
        function_call_json["parameters"]["properties"]["request_heartbeat"] = {
            "type": "boolean",
            "description": "Request an immediate heartbeat after function execution. Set to `True` if you want to send a follow-up message or run a follow-up function.",
        }
        function_call_json["parameters"]["required"].append("request_heartbeat")

    return function_call_json