import json
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from my_project.schemas.enums import RuleType
from my_project.schemas.tool_rule import (
    BasicToolRule,
    SubToolRule,
    ConditionalExecutionRule,
    InitializationRule,
    ExitRule,
)


class RuleValidationException(Exception):
    """Custom exception for errors encountered during tool rule validation."""

    def __init__(self, error_message: str):
        super().__init__(f"RuleValidationException: {error_message}")


class ExecutionRuleManager(BaseModel):
    initialization_rules: List[InitializationRule] = Field(
        default_factory=list,
        description="Rules that initialize tool operations before the main process.",
    )
    process_rules: List[Union[SubToolRule, ConditionalExecutionRule]] = Field(
        default_factory=list,
        description="Primary rules that control tool operation flow and allowed transitions.",
    )
    termination_rules: List[ExitRule] = Field(
        default_factory=list,
        description="Rules that terminate the process if activated.",
    )
    last_used_tool: Optional[str] = Field(
        None, description="Name of the most recently invoked tool."
    )

    def __init__(self, all_rules: List[BasicToolRule], **kwargs):
        super().__init__(**kwargs)
        # Categorize the provided rules into initialization, process, and termination
        for rule in all_rules:
            if rule.type == RuleType.first_run:
                assert isinstance(rule, InitializationRule)
                self.initialization_rules.append(rule)
            elif rule.type == RuleType.restrict_sub_tools:
                assert isinstance(rule, SubToolRule)
                self.process_rules.append(rule)
            elif rule.type == RuleType.conditional:
                assert isinstance(rule, ConditionalExecutionRule)
                self.check_conditional_execution(rule)
                self.process_rules.append(rule)
            elif rule.type == RuleType.terminate_process:
                assert isinstance(rule, ExitRule)
                self.termination_rules.append(rule)

    def track_last_tool_usage(self, tool_name: str):
        """Update internal tracking of the last invoked tool."""
        self.last_used_tool = tool_name

    def fetch_allowed_tools(
        self,
        raise_error_if_empty: bool = False,
        function_response: Optional[str] = None,
    ) -> List[str]:
        """Retrieve a list of permissible tool names based on the previous tool's usage."""
        if self.last_used_tool is None:
            # If no tool has been invoked yet, return initialization rules
            return [rule.tool_name for rule in self.initialization_rules]
        else:
            # Look for a matching rule for the previously used tool
            current_rule = next(
                (
                    rule
                    for rule in self.process_rules
                    if rule.tool_name == self.last_used_tool
                ),
                None,
            )

            if current_rule is None:
                if raise_error_if_empty:
                    raise ValueError(
                        f"No matching rule found for {self.last_used_tool}"
                    )
                return []

            # If it's a conditional execution rule, decide the next tool based on LLM response
            if isinstance(current_rule, ConditionalExecutionRule):
                if not function_response:
                    raise ValueError(
                        "Conditional execution requires a valid LLM response to decide on the next tool."
                    )
                next_tool = self.resolve_conditional_execution(
                    current_rule, function_response
                )
                return [next_tool] if next_tool else []

            return current_rule.children if current_rule.children else []

    def is_termination_rule(self, tool_name: str) -> bool:
        """Determine if the provided tool is a termination tool according to the defined rules."""
        return any(rule.tool_name == tool_name for rule in self.termination_rules)

    def has_subtools(self, tool_name: str) -> bool:
        """Check if a given tool has associated subtools."""
        return any(rule.tool_name == tool_name for rule in self.process_rules)

    def check_conditional_execution(self, rule: ConditionalExecutionRule):
        """
        Ensure that a conditional execution rule is valid.

        Args:
            rule (ConditionalExecutionRule): The conditional rule to validate

        Raises:
            RuleValidationException: If the rule is invalid
        """
        if len(rule.subtool_mapping) == 0:
            raise RuleValidationException(
                "Conditional execution rule must have at least one associated subtool."
            )
        return True

    def resolve_conditional_execution(
        self, rule: ConditionalExecutionRule, function_response: str
    ) -> str:
        """
        Analyze the function's response to determine which subtool should be executed next.

        Args:
            rule (ConditionalExecutionRule): The rule governing conditional execution
            function_response (str): The function's response in JSON format

        Returns:
            str: The name of the subtool to invoke next
        """
        parsed_response = json.loads(function_response)
        output = parsed_response["message"]

        # Try to match the function output with one of the mapping keys
        for key in rule.subtool_mapping:

            # Convert output to the expected type for comparison
            if isinstance(key, bool):
                output_as_type = output.lower() == "true"
            elif isinstance(key, int):
                try:
                    output_as_type = int(output)
                except (ValueError, TypeError):
                    continue
            elif isinstance(key, float):
                try:
                    output_as_type = float(output)
                except (ValueError, TypeError):
                    continue
            else:  # string
                if output == "True" or output == "False":
                    output_as_type = output.lower()
                elif output == "None":
                    output_as_type = None
                else:
                    output_as_type = output

            if output_as_type == key:
                return rule.subtool_mapping[key]

        # Default return value if no match is found
        return rule.default_subtool
