from enum import Enum


class MessageRole(str, Enum):
    """
    Enumeration representing the different roles that a message can have in a given context.

    This is typically used in systems where messages are exchanged between various entities like users, assistants,
    and other components.

    Values:
    - `assistant`: Represents a message sent by an assistant, which is likely an automated response or output
                   generated to assist the user.
    - `user`: Represents a message sent by a user, indicating an input or query from the person interacting with
              the system.
    - `tool`: Represents a message related to a tool. This could be information about the tool's operation, results,
              or status within the system.
    - `function`: Represents a message related to a function call. It might contain details about the function
                  being called, its arguments, or the result of the function execution.
    - `system`: Represents a message from the system itself. This could include instructions, prompts, or other
                system-level information that guides the behavior of the overall process or the other entities.
    """
    assistant = "assistant"
    user = "user"
    tool = "tool"
    function = "function"
    system = "system"


class OptionState(str, Enum):
    """
    Enumeration used to represent the state or option value in a particular context.

    This can be applied in scenarios where a binary or default option needs to be indicated.

    Values:
    - `YES`: Represents an affirmative or enabled state. It indicates that the option is set to a positive or
             selected state.
    - `NO`: Represents a negative or disabled state. It means the option is not selected or is in an inactive state.
    - `DEFAULT`: Represents the default state of an option. This is used when the option has not been explicitly
                 set by the user or the system and should fallback to a predefined default value.
    """
    YES = "yes"
    NO = "no"
    DEFAULT = "default"


class JobStatus(str, Enum):
    """
    Enumeration used to describe the status of a job within a system.

    This is useful for tracking the progress and outcome of tasks or processes that are being executed.

    Values:
    - `created`: Indicates that the job has been created but has not yet started running. It's in an initial state
                  where all the necessary setup might have been done, but the actual processing has not commenced.
    - `running`: Represents that the job is currently in progress. The associated operations or tasks are being
                 actively executed at this stage.
    - `completed`: Signifies that the job has finished successfully. All the required steps have been completed
                    without any errors or issues.
    - `failed`: Means that the job did not complete successfully and encountered an error or problem during its
                execution.
    - `pending`: Suggests that the job is waiting for some external condition or resource before it can start
                 running. This could be due to dependencies on other jobs, waiting for input data, or similar reasons.
    """
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"
    pending = "pending"


class MessageStreamStatus(str, Enum):
    """
    Enumeration used to represent the status of a message stream in a system, likely related to message generation
    or processing steps.

    Values:
    - `done_generation`: Represents that the message generation process has been completed. This indicates that
                         all the messages that were supposed to be generated have been created.
    - `done_step`: Signifies that a particular step in the message processing or generation has been completed.
                   It could be used in a multi-step process to mark the completion of an intermediate stage.
    - `done`: A more general indication that the entire message-related operation, which could involve multiple
              steps like generation and any post-processing, has been finished.
    """
    done_generation = "[DONE_GEN]"
    done_step = "[DONE_STEP]"
    done = "[DONE]"


class ToolRuleType(str, Enum):
    """
    Enumeration used to define different types of tool rules in a system.

    These rules likely govern how tools are used, their order of execution, and any conditional or hierarchical
    relationships between them.

    Values:
    - `run_first`: Also named `InitToolRule`, this indicates that the associated tool should be run first in a
                   sequence or under specific conditions. It defines an initial step or priority for tool execution.
    - `exit_loop`: Also named `TerminalToolRule`, this represents a rule that determines when a loop or iterative
                   process involving tools should stop. It provides the criteria for exiting a tool-related loop.
    - `continue_loop`: Represents a rule that decides whether a loop involving tools should continue or not.
                       It helps in controlling the iterative nature of tool usage.
    - `conditional`: Signifies a rule that is based on certain conditions. The execution of the associated tool
                     might depend on whether specific conditions are met.
    - `constrain_child_tools`: Named `ToolRule`, this implies that it restricts or defines constraints for child
                               tools. It's likely used in a hierarchical tool setup to manage how child tools can
                               operate based on the state or rules of their parent tools.
    - `require_parent_tools`: Represents a rule where the associated tool requires certain parent tools to be
                              available or executed first. It defines a dependency on other tools for its own
                              execution.
    """
    run_first = "InitToolRule"
    exit_loop = "TerminalToolRule"
    continue_loop = "continue_loop"
    conditional = "conditional"
    constrain_child_tools = "ToolRule"
    require_parent_tools = "require_parent_tools"