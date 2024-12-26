from typing import Any, Optional, Union

import humps
from pydantic import BaseModel
import importlib


def generate_composio_tool_wrapper(action_name: str) -> tuple[str, str]:
    """
    Generate a wrapper function for a Composio tool.

    This function creates a string representation of a Python function that instantiates
    a Composio tool based on the provided action name and calls its `func` method with the
    given keyword arguments.

    :param action_name: The name of the action for which to create the tool wrapper.
    :return: A tuple containing the name of the generated function (lowercased action name)
             and the string representation of the wrapper function's code.
    """
    tool_instantiation_str = f"composio_toolset.get_tools(actions=['{action_name}'])[0]"

    func_name = action_name.lower()

    wrapper_function_str = f"""
def {func_name}(**kwargs):
    from composio import Action, App, Tag
    from composio_langchain import ComposioToolSet

    composio_toolset = ComposioToolSet()
    tool = {tool_instantiation_str}
    return tool.func(**kwargs)['data']
    """

    assert_code_gen_compilable(wrapper_function_str)

    return func_name, wrapper_function_str


def generate_langchain_tool_wrapper(
    tool: "LangChainBaseTool", additional_imports_module_attr_map: dict[str, str] = None
) -> tuple[str, str]:
    """
    Generate a wrapper function for a LangChain tool.

    This function constructs a string representation of a Python function that imports
    the necessary classes, instantiates the given LangChain tool, and calls its `_run` method
    with the provided keyword arguments.

    :param tool: The LangChainBaseTool instance for which to create the wrapper.
    :param additional_imports_module_attr_map: A dictionary mapping module names to attribute names
                                                for additional imports needed by the tool (default is None).
    :return: A tuple containing the decamelized name of the tool as the function name and the
             string representation of the wrapper function's code.
    """
    tool_name = tool.__class__.__name__
    import_statement = f"from langchain_community.tools import {tool_name}"
    extra_module_imports = generate_import_code(additional_imports_module_attr_map)

    assert_all_classes_are_imported(tool, additional_imports_module_attr_map)

    tool_instantiation = f"tool = {generate_imported_tool_instantiation_call_str(tool)}"
    run_call = f"return tool._run(**kwargs)"
    func_name = humps.decamelize(tool_name)

    wrapper_function_str = f"""
def {func_name}(**kwargs):
    import importlib
    {import_statement}
    {extra_module_imports}
    {tool_instantiation}
    {run_call}
"""

    assert_code_gen_compilable(wrapper_function_str)

    return func_name, wrapper_function_str


def assert_code_gen_compilable(code_str):
    """
    Check if the given string representing Python code can be compiled.

    If the code has a syntax error, it prints the error message.

    :param code_str: The string containing Python code to be checked.
    """
    try:
        compile(code_str, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")


def assert_all_classes_are_imported(
    tool: Union["LangChainBaseTool"], additional_imports_module_attr_map: dict[str, str]
) -> None:
    """
    Ensure that all required classes for the given tool are imported.

    Compares the set of currently imported classes (based on the tool's class name and
    additional imports specified) with the set of classes required for proper import.
    Raises a RuntimeError if there are missing imports.

    :param tool: The LangChainBaseTool instance.
    :param additional_imports_module_attr_map: A dictionary mapping module names to attribute names
                                                for additional imports (default is None).
    """
    tool_name = tool.__class__.__name__
    current_class_imports = {tool_name}
    if additional_imports_module_attr_map:
        current_class_imports.update(set(additional_imports_module_attr_map.values()))
    required_class_imports = set(find_required_class_names_for_import(tool))

    if not current_class_imports.issuperset(required_class_imports):
        err_msg = f"[ERROR] You are missing module_attr pairs in `additional_imports_module_attr_map`. Currently, you have imports for {current_class_imports}, but the required classes for import are {required_class_imports}"
        print(err_msg)
        raise RuntimeError(err_msg)


def find_required_class_names_for_import(
    obj: Union["LangChainBaseTool", BaseModel]
) -> list[str]:
    """
    Find all the required class names that need to be imported for the given object.

    This function traverses through the object's structure (if it's a Pydantic BaseModel,
    a dictionary, or a list) and collects the unique class names of all the Pydantic models
    found within it.

    :param obj: The object (either a LangChainBaseTool or a Pydantic BaseModel) for which
                to find the required class names.
    :return: A list of required class names as strings.
    """
    class_names = {obj.__class__.__name__}
    queue = [obj]

    while queue:
        curr_obj = queue.pop()

        candidates = []
        if is_base_model(curr_obj):
            fields = dict(curr_obj)
            candidates = list(fields.values())
        elif isinstance(curr_obj, dict):
            candidates = list(curr_obj.values())
        elif isinstance(curr_obj, list):
            candidates = curr_obj

        candidates = filter(lambda x: is_base_model(x), candidates)

        for c in candidates:
            c_name = c.__class__.__name__
            if c_name not in class_names:
                class_names.add(c_name)
                queue.append(c)

    return list(class_names)


def generate_imported_tool_instantiation_call_str(obj: Any) -> Optional[str]:
    """
    Generate a string representation of how to instantiate the given object in Python code.

    Handles different types of objects such as basic Python types (int, float, etc.),
    Pydantic BaseModels, dictionaries, and lists, recursively constructing the appropriate
    code representation. If the object's type is not recognized, it prints a warning and
    returns None.

    :param obj: The object for which to generate the instantiation code string.
    :return: A string representing the instantiation code or None if the object's type
             is not supported.
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return repr(obj)
    elif is_base_model(obj):
        model_name = obj.__class__.__name__
        fields = obj.dict()
        field_assignments = []
        for arg, value in fields.items():
            python_string = generate_imported_tool_instantiation_call_str(value)
            if python_string:
                field_assignments.append(f"{arg}={python_string}")

        assignments = ", ".join(field_assignments)
        return f"{model_name}({assignments})"
    elif isinstance(obj, dict):
        dict_items = []
        for k, v in obj.items():
            python_string = generate_imported_tool_instantiation_call_str(v)
            if python_string:
                dict_items.append(f"{repr(k)}: {python_string}")

        joined_items = ", ".join(dict_items)
        return f"{{{joined_items}}}"
    elif isinstance(obj, list):
        list_items = [generate_imported_tool_instantiation_call_str(v) for v in obj]
        filtered_list_items = list(filter(None, list_items))
        list_items = ", ".join(filtered_list_items)
        return f"[{list_items}]"
    else:
        print(
            f"[WARNING] Skipping parsing unknown class {obj.__class__.__name__} (does not inherit from the Pydantic BaseModel and is not a basic Python type)"
        )
        if obj.__class__.__name__ == "function":
            import inspect

            print(inspect.getsource(obj))

        return None


def is_base_model(obj: Any):
    """
    Check if the given object is an instance of Pydantic's BaseModel or LangChain's equivalent.

    :param obj: The object to check.
    :return: True if the object is an instance of the relevant BaseModel, False otherwise.
    """
    from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel

    return isinstance(obj, BaseModel) or isinstance(obj, LangChainBaseModel)


def generate_import_code(module_attr_map: Optional[dict]) -> str:
    """
    Generate Python code for importing modules and accessing specific attributes from them.

    If the provided module_attr_map is None, an empty string is returned. Otherwise, it
    constructs code lines for each module-attribute pair to import the module and access
    the specified attribute.

    :param module_attr_map: A dictionary mapping module names to attribute names for imports
                            (default is None).
    :return: A string containing the generated import code lines.
    """
    if not module_attr_map:
        return ""

    code_lines = []
    for module, attr in module_attr_map.items():
        module_name = module.split(".")[-1]
        code_lines.append(f"# Load the module\n    {module_name} = importlib.import_module('{module}')")
        code_lines.append(f"# Access the {attr} from the module")
        code_lines.append(f"    {attr} = getattr({module_name}, '{attr}')")
    return "\n".join(code_lines)