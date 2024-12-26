import inspect
from textwrap import dedent
from types import ModuleType
from typing import Dict, List, Optional

from labo.errors import LABOToolCreateError
from labo.functions.schema_generator import generate_schema


def derive_openai_json_schema(source_code: str, name: Optional[str] = None) -> dict:
    """
    Generate an OpenAI JSON schema from the provided source code.

    This function executes the given source code in a controlled environment,
    extracts the last callable function defined in it (excluding built-in special functions),
    and then generates a JSON schema for that function.

    If any errors occur during the process, appropriate LABOToolCreateError exceptions are raised.

    :param source_code: The Python source code as a string.
    :param name: An optional name for the function (used in schema generation, default is None).
    :return: A dictionary representing the generated JSON schema.
    """
    # Create an execution environment with necessary built-in types and globals
    env = {"Optional": Optional, "List": List, "Dict": Dict}
    env.update(globals())

    try:
        # Execute the source code within the created environment
        exec(source_code, env)

        # Find all callable functions in the environment that don't start with '__'
        functions = [f for f in env if callable(env[f]) and not f.startswith("__")]

        if not functions:
            raise LABOToolCreateError("No callable functions found in source code")

        # Get the last callable function found
        func = env[functions[-1]]

        # Check if the function has a docstring
        if not func.__doc__:
            raise LABOToolCreateError(f"Function {func.__name__} missing docstring")

        # Generate the JSON schema for the function
        try:
            schema = generate_schema(func, name=name)
            return schema
        except (TypeError, ValueError) as e:
            raise LABOToolCreateError(f"Schema generation error: {str(e)}")
        except Exception as e:
            raise LABOToolCreateError(f"Unexpected error during schema generation: {str(e)}")

    except Exception as e:
        # Print the stack trace for debugging purposes
        import traceback
        traceback.print_exc()

        # Reraise the LABOToolCreateError with the original error message
        raise LABOToolCreateError(f"Schema generation failed: {str(e)}") from e


def parse_source_code(func) -> str:
    """
    Parse the source code of a given function and clean it up.

    This function uses the 'inspect' module to get the source code of the provided function
    and then applies 'dedent' to remove any common leading whitespace.

    :param func: The function whose source code is to be parsed.
    :return: The cleaned-up source code as a string.
    """
    source_code = dedent(inspect.getsource(func))
    return source_code


def load_function_set(module: ModuleType) -> Dict[str, Dict]:
    """
    Load a set of functions from a given module and generate JSON schemas for them.

    This function iterates over all attributes in the module,
    selects the functions that belong to the module,
    generates a JSON schema for each of them,
    and stores the function, its source code, and the schema in a dictionary.

    If duplicate function names are found or no functions are found in the module,
    appropriate exceptions are raised.

    :param module: The Python module from which to load the functions.
    :return: A dictionary where keys are function names and values are dictionaries containing
             the module's source code, the function itself, and its JSON schema.
    """
    function_dict = {}

    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        # Check if the attribute is a function and belongs to the current module
        if inspect.isfunction(attr) and attr.__module__ == module.__name__:
            if attr_name in function_dict:
                raise ValueError(f"Found a duplicate of function name '{attr_name}'")

            generated_schema = generate_schema(attr)
            function_dict[attr_name] = {
                "module": inspect.getsource(module),
                "python_function": attr,
                "json_schema": generated_schema,
            }

    if not function_dict:
        raise ValueError(f"No functions found in module {module}")

    return function_dict