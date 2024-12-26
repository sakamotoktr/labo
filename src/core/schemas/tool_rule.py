from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from labo.schemas.enums import ToolRuleType
from labo.schemas.labo_base import LABOBase


class BaseToolRule(LABOBase):
    """
    Represents the base structure for tool rules in the system.

    This class inherits from `LABOBase` and serves as a common base for more specific tool rule classes. It
    defines a prefix (`"tool_rule"`) for the unique identifiers of tool rules, which helps in creating
    consistent and identifiable IDs for different tool rules within the system. It also includes two key
    attributes: the name of the tool to which the rule applies (which must exist in the database for the
    user's organization) and the type of the tool rule, which is an enumeration value from `ToolRuleType`.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"tool_rule"`, which is used to prefix the unique identifiers of
                       tool rules. This is used in the process of generating unique IDs for each tool rule
                       instance.
    - `tool_name`: A required string representing the name of the tool that the rule pertains to. This tool
                   must already be present in the database for the user's organization.
    - `type`: An instance of the `ToolRuleType` enum representing the type of the tool rule. Different types
              of tool rules will have different behaviors and additional attributes associated with them.
    """
    __id_prefix__ = "tool_rule"
    tool_name: str = Field(..., description="The name of the tool. Must exist in the database for the user's organization.")
    type: ToolRuleType

class ChildToolRule(BaseToolRule):
    """
    Represents a tool rule that constrains which child tools can be invoked for a given tool.

    This class inherits from `BaseToolRule` and specializes it for the case where the rule is focused on
    limiting the set of child tools that can be used in relation to a particular tool. It sets the `type` of
    the tool rule to `ToolRuleType.constrain_child_tools` and includes a required list of the names of the
    child tools that are permitted to be invoked.

    Attributes:
    - `type`: An instance of the `ToolRuleType` enum set to `ToolRuleType.constrain_child_tools`, indicating
              that this rule is about restricting the available child tools.
    - `children`: A required list of strings representing the names of the children tools that can be invoked.
                  These are the tools that are considered valid to be called within the context of this rule.
    """
    type: ToolRuleType = ToolRuleType.constrain_child_tools
    children: List[str] = Field(..., description="The children tools that can be invoked.")

class TerminalToolRule(BaseToolRule):
    """
    Represents a tool rule that specifies a tool should cause an exit from a loop or sequence of operations.

    This class inherits from `BaseToolRule` and is used to identify a tool that, when invoked, should terminate
    a loop or a specific sequence of tool invocations. It sets the `type` of the tool rule to `ToolRuleType.exit_loop`
    to denote this behavior.

    Attributes:
    - `type`: An instance of the `ToolRuleType` enum set to `ToolRuleType.exit_loop`, indicating that this tool
              is meant to trigger an exit from a loop or ongoing operation.
    """
    type: ToolRuleType = ToolRuleType.exit_loop