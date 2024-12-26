import inspect
import json
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union


class CoreProcessor(ABC):
    @abstractmethod
    def process(
        self,
        inputs: Union[Message, List[Message]],
    ) -> ExecutionMetrics:
        raise NotImplementedError


class Processor(CoreProcessor):
    def __init__(
        self,
        output_handler: Optional[Union[ProcessorInterface, ConsoleOutputHandler]],
        processor_config: ProcessorConfig,
        participant: Participant,
        initial_validation: bool = True,
    ):
        assert isinstance(
            processor_config.knowledge, KnowledgeBase
        ), f"Expected KnowledgeBase type, received: {type(processor_config.knowledge)}"

        self.processor_config = processor_config
        assert isinstance(
            self.processor_config.knowledge, KnowledgeBase
        ), f"Expected KnowledgeBase type, received: {type(self.processor_config.knowledge)}"

        self.participant = participant

        if processor_config.capability_rules:
            for policy in processor_config.capability_rules:
                if not isinstance(policy, FinalCapabilityRule):
                    warnings.warn(
                        "Capability policies are only guaranteed for latest language models supporting structured output."
                    )
                    break

        if processor_config.capability_rules is None:
            processor_config.capability_rules = []

        self.policy_resolver = PolicyResolver(
            capability_rules=processor_config.capability_rules
        )

        self.engine = self.processor_config.language_config.engine
        self.structured_output_enabled = verify_structured_output_support(
            engine=self.engine, capability_rules=processor_config.capability_rules
        )

        self.segment_coordinator = SegmentCoordinator()

        self.output_handler = output_handler

        self.data_coordinator = DataCoordinator()
        self.archive_coordinator = ArchiveCoordinator()
        self.processor_coordinator = ProcessorCoordinator()

        self.initial_validation = initial_validation

        self.resource_warning_triggered = False

        self.previous_capability_result = self.fetch_previous_capability_result()

    def fetch_previous_capability_result(self):
        relevant_messages = self.processor_coordinator.fetch_relevant_messages(
            processor_id=self.processor_config.id, participant=self.participant
        )
        for i in range(len(relevant_messages) - 1, -1, -1):
            entry = relevant_messages[i]
            if entry.role == MessageRole.capability and entry.content:
                try:
                    result_data = json.loads(entry.content)
                    if result_data.get("output"):
                        return result_data["output"]
                except (json.JSONDecodeError, KeyError):
                    raise ValueError(f"Invalid data format in entry: {entry.content}")
        return None

    def refresh_knowledge_if_modified(self, updated_knowledge: KnowledgeBase) -> bool:
        if self.processor_config.knowledge.serialize() != updated_knowledge.serialize():
            for section in self.processor_config.knowledge.list_sections():
                new_value = updated_knowledge.get_section(section).content
                if (
                    new_value
                    != self.processor_config.knowledge.get_section(section).content
                ):
                    section_id = self.processor_config.knowledge.get_section(section).id
                    segment = self.segment_coordinator.modify_segment(
                        segment_id=section_id,
                        segment_changes=SegmentChanges(content=new_value),
                        participant=self.participant,
                    )

            self.processor_config.knowledge = KnowledgeBase(
                segments=[
                    self.segment_coordinator.fetch_segment_by_id(
                        segment.id, participant=self.participant
                    )
                    for segment in self.processor_config.knowledge.get_segments()
                ]
            )

            self.processor_config = self.processor_coordinator.rebuild_instructions(
                processor_id=self.processor_config.id, participant=self.participant
            )

            return True
        return False

    def execute_capability_and_update(
        self,
        capability_name: str,
        capability_params: dict,
        target_capability: Capability,
    ):
        env = {}
        env.update(globals())
        exec(target_capability.implementation, env)
        executable = env[target_capability.schema["name"]]
        specifications = inspect.getfullargspec(executable).annotations
        for name, param in capability_params.items():
            if isinstance(capability_params[name], dict):
                capability_params[name] = specifications[name](
                    **capability_params[name]
                )

        original_knowledge_state = self.processor_config.knowledge.serialize()

        try:
            if (
                capability_name in CORE_CAPABILITIES
                or capability_name in EXTENDED_CORE_CAPABILITIES
            ):
                capability_params["self"] = self
                execution_result = executable(**capability_params)
            else:
                sandbox_execution = CapabilityExecutionEnvironment(
                    capability_name, capability_params, self.participant
                ).execute(processor_config=self.processor_config.__deepcopy__())
                execution_result, updated_config = (
                    sandbox_execution.capability_output,
                    sandbox_execution.processor_config,
                )
                assert (
                    original_knowledge_state
                    == self.processor_config.knowledge.serialize()
                ), "Knowledge base should remain unmodified in sandbox"
                if updated_config is not None:
                    self.refresh_knowledge_if_modified(updated_config.knowledge)
        except Exception as e:
            execution_result = format_error_message(
                capability_name=capability_name,
                error_type=type(e).__name__,
                error_details=str(e),
            )

        return execution_result

    def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: Optional[str] = None,
        first_message: bool = False,
        stream: bool = False,  # TODO move to config?
        empty_response_retry_limit: int = 3,
        backoff_factor: float = 0.5,  # delay multiplier for exponential backoff
        max_delay: float = 10.0,  # max delay between retries
        step_count: Optional[int] = None,
    ) -> ChatCompletionResponse:
        """Get response from LLM API with robust retry mechanism."""

        allowed_tool_names = self.policy_resolver.get_allowed_tool_names(
            last_function_response=self.previous_capability_result
        )
        processor_config_tool_jsons = [t.schema for t in self.processor_config.tools]

        allowed_functions = (
            processor_config_tool_jsons
            if not allowed_tool_names
            else [
                func
                for func in processor_config_tool_jsons
                if func["name"] in allowed_tool_names
            ]
        )

        # For the first message, force the initial tool if one is specified
        force_tool_call = None
        if (
            step_count is not None
            and step_count == 0
            and not self.structured_output_enabled
            and len(self.policy_resolver.init_tool_rules) > 0
        ):
            force_tool_call = self.policy_resolver.init_tool_rules[0].tool_name
        # Force a tool call if exactly one tool is specified
        elif step_count is not None and step_count > 0 and len(allowed_tool_names) == 1:
            force_tool_call = allowed_tool_names[0]
        for attempt in range(1, empty_response_retry_limit + 1):
            try:
                response = create(
                    llm_config=self.processor_config.language_config,
                    messages=message_sequence,
                    user_id=self.processor_config.created_by_id,
                    functions=allowed_functions,
                    # functions_python=self.functions_python, do we need this?
                    function_call=function_call,
                    first_message=first_message,
                    force_tool_call=force_tool_call,
                    stream=stream,
                    stream_interface=self.output_handler,
                )

                # These bottom two are retryable
                if len(response.choices) == 0 or response.choices[0] is None:
                    raise ValueError(f"API call returned an empty message: {response}")

                if response.choices[0].finish_reason not in [
                    "stop",
                    "function_call",
                    "tool_calls",
                ]:
                    if response.choices[0].finish_reason == "length":
                        # This is not retryable, hence RuntimeError v.s. ValueError
                        raise RuntimeError(
                            "Finish reason was length (maximum context length)"
                        )
                    else:
                        raise ValueError(
                            f"Bad finish reason from API: {response.choices[0].finish_reason}"
                        )

                return response

            except ValueError as ve:
                if attempt >= empty_response_retry_limit:
                    warnings.warn(f"Retry limit reached. Final error: {ve}")
                    raise Exception(
                        f"Retries exhausted and no valid response received. Final error: {ve}"
                    )
                else:
                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_delay)
                    warnings.warn(
                        f"Attempt {attempt} failed: {ve}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)

            except Exception as e:
                # For non-retryable errors, exit immediately
                raise e

        raise Exception("Retries exhausted and no valid response received.")

    def _handle_ai_response(
        self,
        response_message: ChatCompletionMessage,  # TODO should we eventually move the Message creation outside of this function?
        override_tool_call_id: bool = False,
        # If we are streaming, we needed to create a Message ID ahead of time,
        # and now we want to use it in the creation of the Message object
        # TODO figure out a cleaner way to do this
        response_message_id: Optional[str] = None,
    ) -> Tuple[List[Message], bool, bool]:
        """Handles parsing and function execution"""

        # Hacky failsafe for now to make sure we didn't implement the streaming Message ID creation incorrectly
        if response_message_id is not None:
            assert response_message_id.startswith("message-"), response_message_id

        messages = []  # append these to the history when done
        function_name = None

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (
            response_message.tool_calls is not None
            and len(response_message.tool_calls) > 0
        ):
            if response_message.function_call:
                raise DeprecationWarning(response_message)
            if (
                response_message.tool_calls is not None
                and len(response_message.tool_calls) > 1
            ):
                # raise NotImplementedError(f">1 tool call not supported")
                # TODO eventually support sequential tool calling
                printd(
                    f">1 tool call not supported, using index=0 only\n{response_message.tool_calls}"
                )
                response_message.tool_calls = [response_message.tool_calls[0]]
            assert (
                response_message.tool_calls is not None
                and len(response_message.tool_calls) > 0
            )

            # generate UUID for tool call
            if override_tool_call_id or response_message.function_call:
                warnings.warn(
                    "Overriding the tool call can result in inconsistent tool call IDs during streaming"
                )
                tool_call_id = get_tool_call_id()  # needs to be a string for JSON
                response_message.tool_calls[0].id = tool_call_id
            else:
                tool_call_id = response_message.tool_calls[0].id
                assert tool_call_id is not None  # should be defined

            # only necessary to add the tool_cal_id to a function call (antipattern)
            # response_message_dict = response_message.model_dump()
            # response_message_dict["tool_call_id"] = tool_call_id

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                # TODO should probably just overwrite the fields?
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            printd(f"Function call message: {messages[-1]}")

            nonnull_content = False
            if response_message.content:
                # The content if then internal monologue, not chat
                self.output_handler.internal_monologue(
                    response_message.content, msg_obj=messages[-1]
                )
                # Flag to avoid printing a duplicate if inner thoughts get popped from the function call
                nonnull_content = True

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            function_call = (
                response_message.function_call
                if response_message.function_call is not None
                else response_message.tool_calls[0].function
            )

            # Get the name of the function
            function_name = function_call.name
            printd(
                f"Request to call function {function_name} with tool_call_id: {tool_call_id}"
            )

            # Failure case 1: function name is wrong (not in processor_config.tools)
            target_capability = None
            for t in self.processor_config.tools:
                if t.name == function_name:
                    target_capability = t

            if not target_capability:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.processor_config.id,
                        user_id=self.processor_config.created_by_id,
                        model=self.engine,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.output_handler.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = function_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.processor_config.id,
                        user_id=self.processor_config.created_by_id,
                        model=self.engine,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.output_handler.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # Check if inner thoughts is in the function call arguments (possible apparently if you are using Azure)
            if "inner_thoughts" in function_args:
                response_message.content = function_args.pop("inner_thoughts")
            # The content if then internal monologue, not chat
            if response_message.content and not nonnull_content:
                self.output_handler.internal_monologue(
                    response_message.content, msg_obj=messages[-1]
                )

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)

            # Edge case: heartbeat_request is returned as a stringified boolean, we will attempt to parse:
            if (
                isinstance(heartbeat_request, str)
                and heartbeat_request.lower().strip() == "true"
            ):
                heartbeat_request = True

            if not isinstance(heartbeat_request, bool) or heartbeat_request is None:
                printd(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = False

            # Failure case 3: function failed during execution
            # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
            #       this is because the function/tool role message is only created once the function/tool has executed/returned
            self.output_handler.function_message(
                f"Running {function_name}({function_args})", msg_obj=messages[-1]
            )
            try:
                # handle tool execution (sandbox) and state updates
                function_response = self.execute_capability_and_update(
                    function_name, function_args, target_capability
                )

                # handle trunction
                if function_name in [
                    "conversation_search",
                    "conversation_search_date",
                    "archival_memory_search",
                ]:
                    # with certain functions we rely on the paging mechanism to handle overflow
                    truncate = False
                else:
                    # but by default, we add a truncation safeguard to prevent bad functions from
                    # overflow the agent context window
                    truncate = True

                # get the function response limit
                return_char_limit = target_capability.return_char_limit
                function_response_string = validate_function_response(
                    function_response,
                    return_char_limit=return_char_limit,
                    truncate=truncate,
                )
                function_args.pop("self", None)
                function_response = package_function_response(
                    True, function_response_string
                )
                function_failed = False
            except Exception as e:
                function_args.pop("self", None)
                # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                error_msg = get_friendly_error_msg(
                    function_name=function_name,
                    exception_name=type(e).__name__,
                    exception_message=str(e),
                )
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                printd(error_msg_user)
                function_response = package_function_response(False, error_msg)
                self.previous_capability_result = function_response
                # TODO: truncate error message somehow
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.processor_config.id,
                        user_id=self.processor_config.created_by_id,
                        model=self.engine,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.output_handler.function_message(
                    f"Ran {function_name}({function_args})", msg_obj=messages[-1]
                )
                self.output_handler.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # Step 4: check if function response is an error
            if function_response_string.startswith(ERROR_MESSAGE_PREFIX):
                function_response = package_function_response(
                    False, function_response_string
                )
                # TODO: truncate error message somehow
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.processor_config.id,
                        user_id=self.processor_config.created_by_id,
                        model=self.engine,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.output_handler.function_message(
                    f"Ran {function_name}({function_args})", msg_obj=messages[-1]
                )
                self.output_handler.function_message(
                    f"Error: {function_response_string}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 5: send the info on the function call and function response to GPT
            messages.append(
                Message.dict_to_message(
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        "tool_call_id": tool_call_id,
                    },
                )
            )  # extend conversation with function response
            self.output_handler.function_message(
                f"Ran {function_name}({function_args})", msg_obj=messages[-1]
            )
            self.output_handler.function_message(
                f"Success: {function_response_string}", msg_obj=messages[-1]
            )
            self.previous_capability_result = function_response

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.output_handler.internal_monologue(
                response_message.content, msg_obj=messages[-1]
            )
            heartbeat_request = False
            function_failed = False

        # rebuild knowledge
        # TODO: @charles please check this
        self.processor_config = self.processor_coordinator.rebuild_instructions(
            processor_id=self.processor_config.id, participant=self.participant
        )

        # Update PolicyResolver state with last called function
        self.policy_resolver.update_tool_usage(function_name)
        # Update heartbeat request according to provided tool rules
        if self.policy_resolver.has_children_tools(function_name):
            heartbeat_request = True
        elif self.policy_resolver.is_terminal_tool(function_name):
            heartbeat_request = False

        return messages, heartbeat_request, function_failed

    def process(
        self,
        inputs: Union[Message, List[Message]],
        # additional args
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        **kwargs,
    ) -> ExecutionMetrics:
        """Run Processor.process in a loop, handling chaining via heartbeat requests and function failures"""
        next_input_message = inputs if isinstance(inputs, list) else [inputs]
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0
        while True:
            kwargs["first_message"] = False
            kwargs["step_count"] = step_count
            step_response = self.inner_process(
                inputs=next_input_message,
                **kwargs,
            )

            heartbeat_request = step_response.heartbeat_request
            function_failed = step_response.function_failed
            token_warning = step_response.in_context_memory_warning
            usage = step_response.usage

            step_count += 1
            total_usage += usage
            counter += 1
            self.output_handler.step_complete()

            # logger.debug("Saving processor state")
            # save updated state
            save_processor(self)

            # Chain stops
            if not chaining:
                printd("No chaining, stopping after one step")
                break
            elif max_chaining_steps is not None and counter > max_chaining_steps:
                printd(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif token_warning:
                assert self.processor_config.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_token_limit_warning(),
                    },
                )
                continue  # always chain
            elif function_failed:
                assert self.processor_config.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_heartbeat(FUNC_FAILED_HEARTBEAT_MESSAGE),
                    },
                )
                continue  # always chain
            elif heartbeat_request:
                assert self.processor_config.created_by_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_heartbeat(REQ_HEARTBEAT_MESSAGE),
                    },
                )
                continue  # always chain
            # LABO no-op / yield
            else:
                break

        return ExecutionMetrics(**total_usage.model_dump(), step_count=step_count)

    def inner_process(
        self,
        inputs: Union[Message, List[Message]],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        stream: bool = False,  # TODO move to config?
        step_count: Optional[int] = None,
    ) -> ProcessorStepResponse:
        """Runs a single step in the processor loop (generates at most one LLM call)"""

        try:

            # Step 0: update core knowledge
            # only pulling latest segment data if shared knowledge is being used
            current_persisted_knowledge = KnowledgeBase(
                segments=[
                    self.segment_coordinator.fetch_segment_by_id(
                        segment.id, participant=self.participant
                    )
                    for segment in self.processor_config.knowledge.get_segments()
                ]
            )  # read segments from DB
            self.refresh_knowledge_if_modified(current_persisted_knowledge)

            # Step 1: add user message
            if isinstance(inputs, Message):
                inputs = [inputs]

            if not all(isinstance(m, Message) for m in inputs):
                raise ValueError(
                    f"inputs should be a Message or a list of Message, got {type(inputs)}"
                )

            relevant_messages = self.processor_coordinator.fetch_relevant_messages(
                processor_id=self.processor_config.id, participant=self.participant
            )
            input_message_sequence = relevant_messages + inputs

            if (
                len(input_message_sequence) > 1
                and input_message_sequence[-1].role != "user"
            ):
                printd(
                    f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue"
                )

            # Step 2: send the conversation and available functions to the LLM
            response = self._get_ai_reply(
                message_sequence=input_message_sequence,
                first_message=first_message,
                stream=stream,
                step_count=step_count,
            )

            # Step 3: check if LLM wanted to call a function
            # (if yes) Step 4: call the function
            # (if yes) Step 5: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message.model_copy()  # TODO why are we copying here?
            all_response_messages, heartbeat_request, function_failed = (
                self._handle_ai_response(
                    response_message,
                    # TODO this is kind of hacky, find a better way to handle this
                    # the only time we set up message creation ahead of time is when streaming is on
                    response_message_id=response.id if stream else None,
                )
            )

            # Step 6: extend the message history
            if len(inputs) > 0:
                all_new_messages = inputs + all_response_messages
            else:
                all_new_messages = all_response_messages

            # Check the knowledge pressure and potentially issue a knowledge pressure warning
            current_total_tokens = response.usage.total_tokens
            active_knowledge_warning = False

            # We can't do summarize logic properly if context_window is undefined
            if self.processor_config.language_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(
                    f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
                )
                print(f"{self.processor_config}")
                self.processor_config.language_config.context_window = (
                    LLM_MAX_TOKENS[self.engine]
                    if (self.engine is not None and self.engine in LLM_MAX_TOKENS)
                    else LLM_MAX_TOKENS["DEFAULT"]
                )

            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(
                self.processor_config.language_config.context_window
            ):
                printd(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.processor_config.language_config.context_window)}"
                )

                # Only deliver the alert if we haven't already (this period)
                if not self.resource_warning_triggered:
                    active_knowledge_warning = True
                    self.resource_warning_triggered = (
                        True  # it's up to the outer loop to handle this
                    )

            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.processor_config.language_config.context_window)}"
                )

            self.processor_config = (
                self.processor_coordinator.append_to_relevant_messages(
                    all_new_messages,
                    processor_id=self.processor_config.id,
                    participant=self.participant,
                )
            )

            return ProcessorStepResponse(
                messages=all_new_messages,
                heartbeat_request=heartbeat_request,
                function_failed=function_failed,
                in_context_memory_warning=active_knowledge_warning,
                usage=response.usage,
            )

        except Exception as e:
            printd(f"process() failed\ninputs = {inputs}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                printd(
                    f"context window exceeded with limit {self.processor_config.language_config.context_window}, running summarizer to trim messages"
                )
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try process again
                return self.inner_process(
                    inputs=inputs,
                    first_message=first_message,
                    first_message_retry_limit=first_message_retry_limit,
                    skip_verify=skip_verify,
                    stream=stream,
                )

            else:
                printd(f"process() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def process_user_message(
        self, user_message_str: str, **kwargs
    ) -> ProcessorStepResponse:
        """Takes a basic user message string, turns it into a stringified JSON with extra metadata, then sends it to the processor

        Example:
        -> user_message_str = 'hi'
        -> {'message': 'hi', 'type': 'user_message', ...}
        -> json.dumps(...)
        -> processor.process(inputs=[Message(role='user', text=...)])
        """
        # Wrap with metadata, dumps to JSON
        assert user_message_str and isinstance(
            user_message_str, str
        ), f"user_message_str should be a non-empty string, got {type(user_message_str)}"
        user_message_json_str = package_user_message(user_message_str)

        # Validate JSON via save/load
        user_message = validate_json(user_message_json_str)
        cleaned_user_message_text, name = strip_name_field_from_user_message(
            user_message
        )

        # Turn into a dict
        openai_message_dict = {
            "role": "user",
            "content": cleaned_user_message_text,
            "name": name,
        }

        # Create the associated Message object (in the database)
        assert self.processor_config.created_by_id is not None, "User ID is not set"
        user_message = Message.dict_to_message(
            agent_id=self.processor_config.id,
            user_id=self.processor_config.created_by_id,
            model=self.engine,
            openai_message_dict=openai_message_dict,
            # created_at=timestamp,
        )

        return self.inner_process(inputs=[user_message], **kwargs)

    def summarize_messages_inplace(
        self, cutoff=None, preserve_last_N_messages=True, disallow_tool_as_first=True
    ):
        relevant_messages = self.processor_coordinator.fetch_relevant_messages(
            processor_id=self.processor_config.id, participant=self.participant
        )
        relevant_messages_openai = [m.to_openai_dict() for m in relevant_messages]

        if relevant_messages_openai[0]["role"] != "system":
            raise RuntimeError(
                f"relevant_messages_openai[0] should be system (instead got {relevant_messages_openai[0]})"
            )

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in relevant_messages_openai]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(
            message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC
        )
        candidate_messages_to_summarize = relevant_messages_openai[1:]
        token_counts = token_counts[1:]

        if preserve_last_N_messages:
            candidate_messages_to_summarize = candidate_messages_to_summarize[
                :-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST
            ]
            token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]

        printd(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        printd(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        printd(f"token_counts={token_counts}")
        printd(f"message_buffer_token_count={message_buffer_token_count}")
        printd(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        printd(
            f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}"
        )

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise ContextWindowExceededError(
                "Not enough messages to compress for summarization",
                details={
                    "num_candidate_messages": len(candidate_messages_to_summarize),
                    "num_total_messages": len(relevant_messages_openai),
                    "preserve_N": MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
                },
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            printd(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if relevant_messages_openai[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if relevant_messages_openai[new_cutoff]["role"] == "user":
                    printd(
                        f"Shifted cutoff {new_cutoff} is still a 'user', ignoring..."
                    )
                cutoff = new_cutoff
        except IndexError:
            pass

        # Make sure the cutoff isn't on a 'tool' or 'function'
        if disallow_tool_as_first:
            while relevant_messages_openai[cutoff]["role"] in [
                "tool",
                "function",
            ] and cutoff < len(relevant_messages_openai):
                printd(f"Selected cutoff {cutoff} was a 'tool', shifting one...")
                cutoff += 1

        message_sequence_to_summarize = relevant_messages[
            1:cutoff
        ]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise ContextWindowExceededError(
                "Not enough messages to compress for summarization after determining cutoff",
                details={
                    "num_candidate_messages": len(message_sequence_to_summarize),
                    "num_total_messages": len(relevant_messages_openai),
                    "preserve_N": MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
                },
            )
        else:
            printd(
                f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(relevant_messages)}"
            )

        # We can't do summarize logic properly if context_window is undefined
        if self.processor_config.language_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(
                f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
            )
            print(f"{self.processor_config}")
            self.processor_config.language_config.context_window = (
                LLM_MAX_TOKENS[self.engine]
                if (self.engine is not None and self.engine in LLM_MAX_TOKENS)
                else LLM_MAX_TOKENS["DEFAULT"]
            )

        summary = summarize_messages(
            processor_config=self.processor_config,
            message_sequence_to_summarize=message_sequence_to_summarize,
        )
        printd(f"Got summary: {summary}")

        # Metadata that's useful for the processor to see
        all_time_message_count = self.data_coordinator.size(
            processor_id=self.processor_config.id, participant=self.participant
        )
        remaining_message_count = len(relevant_messages_openai[cutoff:])
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(
            summary, summary_message_count, hidden_message_count, all_time_message_count
        )
        printd(f"Packaged into message: {summary_message}")

        prior_len = len(relevant_messages_openai)
        self.processor_config = self.processor_coordinator.trim_older_relevant_messages(
            cutoff, processor_id=self.processor_config.id, participant=self.participant
        )
        packed_summary_message = {"role": "user", "content": summary_message}
        self.processor_config = self.processor_coordinator.prepend_to_relevant_messages(
            messages=[
                Message.dict_to_message(
                    agent_id=self.processor_config.id,
                    user_id=self.processor_config.created_by_id,
                    model=self.engine,
                    openai_message_dict=packed_summary_message,
                )
            ],
            processor_id=self.processor_config.id,
            participant=self.participant,
        )

        # reset alert
        self.resource_warning_triggered = False

        printd(
            f"Ran summarizer, messages length {prior_len} -> {len(relevant_messages_openai)}"
        )

    def add_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def remove_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def migrate_embedding(self, embedding_config: EmbeddingConfig):
        """Migrate the processor to a new embedding"""
        # TODO: archival knowledge

        # TODO: recall knowledge
        raise NotImplementedError()

    def get_context_window(self) -> ContextWindowOverview:
        """Get the context window of the processor"""

        system_prompt = (
            self.processor_config.system
        )  # TODO is this the current system or the initial system?
        num_tokens_system = count_tokens(system_prompt)
        core_knowledge = self.processor_config.knowledge.serialize()
        num_tokens_core_knowledge = count_tokens(core_knowledge)

        # Grab the relevant messages
        # conversion of messages to OpenAI dict format, which is passed to the token counter
        relevant_messages = self.processor_coordinator.fetch_relevant_messages(
            processor_id=self.processor_config.id, participant=self.participant
        )
        relevant_messages_openai = [m.to_openai_dict() for m in relevant_messages]

        # Check if there's a summary message in the message queue
        if (
            len(relevant_messages) > 1
            and relevant_messages[1].role == MessageRole.user
            and isinstance(relevant_messages[1].content, str)
            # TODO remove hardcoding
            and "The following is a summary of the previous "
            in relevant_messages[1].content
        ):
            # Summary message exists
            assert relevant_messages[1].content is not None
            summary_knowledge = relevant_messages[1].content
            num_tokens_summary_knowledge = count_tokens(relevant_messages[1].content)
            # with a summary message, the real messages start at index 2
            num_tokens_messages = (
                num_tokens_from_messages(
                    messages=relevant_messages_openai[2:], model=self.engine
                )
                if len(relevant_messages_openai) > 2
                else 0
            )

        else:
            summary_knowledge = None
            num_tokens_summary_knowledge = 0
            # with no summary message, the real messages start at index 1
            num_tokens_messages = (
                num_tokens_from_messages(
                    messages=relevant_messages_openai[1:], model=self.engine
                )
                if len(relevant_messages_openai) > 1
                else 0
            )

        processor_coordinator_archive_size = self.processor_coordinator.archive_size(
            participant=self.participant, processor_id=self.processor_config.id
        )
        data_coordinator_size = self.data_coordinator.size(
            participant=self.participant, processor_id=self.processor_config.id
        )
        external_knowledge_summary = compile_knowledge_metadata_block(
            knowledge_edit_timestamp=get_utc_time(),
            previous_message_count=self.data_coordinator.size(
                participant=self.participant, processor_id=self.processor_config.id
            ),
            archival_knowledge_size=self.processor_coordinator.archive_size(
                participant=self.participant, processor_id=self.processor_config.id
            ),
        )
        num_tokens_external_knowledge_summary = count_tokens(external_knowledge_summary)

        # tokens taken up by function definitions
        processor_config_tool_jsons = [t.schema for t in self.processor_config.tools]
        if processor_config_tool_jsons:
            available_functions_definitions = [
                ChatCompletionRequestTool(type="function", function=f)
                for f in processor_config_tool_jsons
            ]
            num_tokens_available_functions_definitions = num_tokens_from_functions(
                functions=processor_config_tool_jsons, model=self.engine
            )
        else:
            available_functions_definitions = []
            num_tokens_available_functions_definitions = 0

        num_tokens_used_total = (
            num_tokens_system  # system prompt
            + num_tokens_available_functions_definitions  # function definitions
            + num_tokens_core_knowledge  # core knowledge
            + num_tokens_external_knowledge_summary  # metadata (statistics) about recall/archival
            + num_tokens_summary_knowledge  # summary of ongoing conversation
            + num_tokens_messages  # tokens taken by messages
        )
        assert isinstance(num_tokens_used_total, int)

        return ContextWindowOverview(
            # context window breakdown (in messages)
            num_messages=len(relevant_messages),
            num_archival_knowledge=processor_coordinator_archive_size,
            num_recall_knowledge=data_coordinator_size,
            num_tokens_external_knowledge_summary=num_tokens_external_knowledge_summary,
            external_knowledge_summary=external_knowledge_summary,
            # top-level information
            context_window_size_max=self.processor_config.language_config.context_window,
            context_window_size_current=num_tokens_used_total,
            # context window breakdown (in tokens)
            num_tokens_system=num_tokens_system,
            system_prompt=system_prompt,
            num_tokens_core_knowledge=num_tokens_core_knowledge,
            core_knowledge=core_knowledge,
            num_tokens_summary_knowledge=num_tokens_summary_knowledge,
            summary_knowledge=summary_knowledge,
            num_tokens_messages=num_tokens_messages,
            messages=relevant_messages,
            # related to functions
            num_tokens_functions_definitions=num_tokens_available_functions_definitions,
            functions_definitions=available_functions_definitions,
        )

    def count_tokens(self) -> int:
        """Count the tokens in the current context window"""
        context_window_breakdown = self.get_context_window()
        return context_window_breakdown.context_window_size_current


def save_processor(processor: Processor):
    """Save processor to metadata store"""
    processor_config = processor.processor_config
    assert isinstance(
        processor_config.knowledge, KnowledgeBase
    ), f"Knowledge is not a KnowledgeBase object: {type(processor_config.knowledge)}"

    # TODO: move this to processor coordinator
    # TODO: Completely strip out metadata
    # convert to persisted model
    processor_coordinator = ProcessorCoordinator()
    update_processor = UpdateProcessor(
        name=processor_config.name,
        tool_ids=[t.id for t in processor_config.tools],
        source_ids=[s.id for s in processor_config.sources],
        segment_ids=[b.id for b in processor_config.knowledge.segments],
        tags=processor_config.tags,
        system=processor_config.system,
        capability_rules=processor_config.capability_rules,
        language_config=processor_config.language_config,
        embedding_config=processor_config.embedding_config,
        message_ids=processor_config.message_ids,
        description=processor_config.description,
        metadata_=processor_config.metadata_,
    )
    processor_coordinator.update_processor(
        processor_id=processor_config.id,
        processor_update=update_processor,
        participant=processor.participant,
    )


def strip_name_field_from_user_message(
    user_message_text: str,
) -> Tuple[str, Optional[str]]:
    """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        # Special handling for AutoGen messages with 'name' field
        # Treat 'name' as a special field
        # If it exists in the input message, elevate it to the 'message' level
        name = user_message_json.pop("name", None)
        clean_message = json_dumps(user_message_json)
        return clean_message, name

    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")
        raise e


def validate_json(user_message_text: str) -> str:
    """Make sure that the user input message is valid JSON"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        user_message_json_val = json_dumps(user_message_json)
        return user_message_json_val
    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
        raise e
