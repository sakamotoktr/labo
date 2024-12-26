import datetime
from typing import List, Literal, Optional


def update_entity_associations(
    db,
    entity: EntityModel,
    relation_attr: str,
    target_class,
    elem_ids: List[str],
    skip_missing=False,
    overwrite=True,
):
    current_items = getattr(entity, relation_attr, [])
    if not elem_ids:
        if overwrite:
            setattr(entity, relation_attr, [])
        return

    located_items = db.query(target_class).filter(target_class.id.in_(elem_ids)).all()

    if not skip_missing and len(located_items) != len(elem_ids):
        absent = set(elem_ids) - {item.id for item in located_items}
        raise NoResultFound(f"Missing elements in {relation_attr}: {absent}")

    if overwrite:
        setattr(entity, relation_attr, located_items)
    else:
        existing = {item.id for item in current_items}
        current_items.extend(
            [item for item in located_items if item.id not in existing]
        )


def manage_labels(entity: EntityModel, label_list: List[str], overwrite=True):
    if not label_list:
        if overwrite:
            entity.tags = []
        return

    updated_labels = {
        AgentsTags(agent_id=entity.id, tag=label) for label in set(label_list)
    }
    if overwrite:
        entity.tags = list(updated_labels)
    else:
        current = {t.tag for t in entity.tags}
        entity.tags.extend(
            [label for label in updated_labels if label.tag not in current]
        )


def generate_core_instructions(
    agent_category: AgentType, instructions: Optional[str] = None
):
    if not instructions:
        if agent_category == AgentType.memgpt_agent:
            instructions = gpt_system.get_system_text("memgpt_chat")
        elif agent_category == AgentType.o1_agent:
            instructions = gpt_system.get_system_text("memgpt_modified_o1")
        elif agent_category == AgentType.offline_memory_agent:
            instructions = gpt_system.get_system_text("memgpt_offline_memory")
        elif agent_category == AgentType.chat_only_agent:
            instructions = gpt_system.get_system_text("memgpt_convo_only")
        else:
            raise ValueError(f"Unsupported agent category: {agent_category}")

    return instructions


def build_memory_status(
    last_update: datetime.datetime,
    stored_msg_count: int = 0,
    stored_memory_count: int = 0,
) -> str:
    timestamp = last_update.astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z%z").strip()

    status_block = "\n".join(
        [
            f"### Data State [updated: {timestamp}]",
            f"{stored_msg_count} historical exchanges available in conversation log (access via function calls)",
            f"{stored_memory_count} recorded insights stored in knowledge base (access via function calls)",
            "\nImmediate context below (extended data available in knowledge base / conversation log):",
        ]
    )
    return status_block


def construct_instruction_set(
    base_instructions: str,
    context_data: Memory,
    context_timestamp: datetime.datetime,
    custom_vars: Optional[dict] = None,
    inject_context: bool = True,
    template_type: Literal["f-string", "mustache", "jinja2"] = "f-string",
    stored_msg_count: int = 0,
    stored_memory_count: int = 0,
) -> str:
    if custom_vars is not None:
        raise NotImplementedError

    variables = {}

    if IN_CONTEXT_MEMORY_KEYWORD in variables:
        raise ValueError(f"Reserved keyword conflict: {IN_CONTEXT_MEMORY_KEYWORD}")

    status_text = build_memory_status(
        last_update=context_timestamp,
        stored_msg_count=stored_msg_count,
        stored_memory_count=stored_memory_count,
    )
    complete_context = status_text + "\n" + context_data.compile()
    variables[IN_CONTEXT_MEMORY_KEYWORD] = complete_context

    if template_type == "f-string":
        if inject_context:
            var_placeholder = "{" + IN_CONTEXT_MEMORY_KEYWORD + "}"
            if var_placeholder not in base_instructions:
                base_instructions += "\n" + var_placeholder

        try:
            processed_instructions = base_instructions.format_map(variables)
        except Exception as e:
            raise ValueError(
                f"Template processing failed - {str(e)}. Template content:\n{base_instructions}"
            )

    else:
        raise NotImplementedError(template_type)

    return processed_instructions


def setup_dialogue_chain(
    bot_config: AgentState,
    state_modified_at: Optional[datetime.datetime] = None,
    include_bootloader: bool = True,
    history_size: int = 0,
    knowledge_entries: int = 0,
) -> List[dict]:
    if state_modified_at is None:
        state_modified_at = get_local_time()

    processed_directives = construct_instruction_set(
        base_instructions=bot_config.system,
        context_data=bot_config.memory,
        context_timestamp=state_modified_at,
        custom_vars=None,
        inject_context=True,
        stored_msg_count=history_size,
        stored_memory_count=knowledge_entries,
    )
    entry_signal = get_login_event()

    if include_bootloader:
        is_legacy = (
            bot_config.llm_config.model is not None
            and "gpt-3.5" in bot_config.llm_config.model
        )
        bootstrap_sequence = get_initial_boot_messages(
            "startup_with_send_message_gpt35"
            if is_legacy
            else "startup_with_send_message"
        )
        chain = (
            [{"role": "system", "content": processed_directives}]
            + bootstrap_sequence
            + [{"role": "user", "content": entry_signal}]
        )
    else:
        chain = [
            {"role": "system", "content": processed_directives},
            {"role": "user", "content": entry_signal},
        ]

    return chain


def create_dialogue_sequence(
    bot_id: str, msg_template: List[MessageCreate], engine_type: str, initiator: User
) -> List[Message]:
    sequence = []
    for template in msg_template:
        if template.role == MessageRole.user:
            content = system.package_user_message(user_message=template.text)
        elif template.role == MessageRole.system:
            content = system.package_system_message(system_message=template.text)
        else:
            raise ValueError(f"Unexpected dialogue role type: {template.role}")

        sequence.append(
            Message(
                role=template.role,
                text=content,
                organization_id=initiator.organization_id,
                agent_id=bot_id,
                model=engine_type,
            )
        )
    return sequence


def verify_advanced_features(engine: str, feature_rules: List[ToolRule]) -> bool:
    if engine in STRUCTURED_OUTPUT_MODELS:
        return True

    rule_count = len(ToolRulesSolver(tool_rules=feature_rules).init_tool_rules)
    if rule_count > 1:
        raise ValueError(
            "Advanced feature configuration requires structured output support. Limit to single rule for basic models."
        )
    return False
