from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union


class DialogProcessor(Agent):
    def __init__(
        self,
        interface_handler: AgentInterface,
        processor_state: AgentState,
        participant: User,
        initial_validation_single: bool = False,
        continuous_memory_refresh: bool = True,
        dialog_context_size: int = 2000,
    ):
        super().__init__(interface_handler, processor_state, participant)
        self.initial_validation_single = initial_validation_single
        self.continuous_memory_refresh = continuous_memory_refresh
        self.background_processor = None
        self.dialog_context_size = dialog_context_size

    def process_interaction(
        self,
        interaction_data: Union[Message, List[Message]],
        enable_chain: bool = True,
        chain_depth_limit: Optional[int] = None,
        **extra_params,
    ) -> LABOUsageStatistics:
        interaction_stats = super().step(
            messages=interaction_data,
            chaining=enable_chain,
            max_chaining_steps=chain_depth_limit,
            **extra_params,
        )

        if self.continuous_memory_refresh:

            def establish_background_processor():
                session = create_client()
                if self.background_processor:
                    session.delete_agent(agent_id=self.background_processor.id)
                    self.background_processor = None

                participant_context = self.processor_state.memory.get_block(
                    "chat_agent_human"
                )
                system_context = self.processor_state.memory.get_block(
                    "chat_agent_persona"
                )

                background_system = Block(
                    name="background_processor_identity",
                    label="background_processor_identity",
                    value=get_persona_text("background_processor_identity"),
                    limit=2000,
                )

                updated_participant_context = Block(
                    name="dialog_participant_current",
                    label="dialog_participant_current",
                    value=participant_context.value,
                    limit=2000,
                )

                updated_system_context = Block(
                    name="dialog_system_current",
                    label="dialog_system_current",
                    value=system_context.value,
                    limit=2000,
                )

                relevant_messages = self.agent_manager.get_in_context_messages(
                    agent_id=self.processor_state.id, actor=self.participant
                )

                current_dialog = "".join([str(msg) for msg in relevant_messages[3:]])[
                    -self.dialog_context_size :
                ]

                dialog_snapshot = Block(
                    name="current_interaction_state",
                    label="current_interaction_state",
                    value=current_dialog,
                    limit=self.dialog_context_size,
                )

                enhanced_memory = BasicBlockMemory(
                    blocks=[
                        background_system,
                        participant_context,
                        system_context,
                        updated_participant_context,
                        updated_system_context,
                        dialog_snapshot,
                    ]
                )

                self.background_processor = session.create_agent(
                    name="background_memory_processor",
                    agent_type=AgentType.offline_memory_agent,
                    system=gpt_system.get_system_text("enhanced_memory_processor"),
                    memory=enhanced_memory,
                    llm_config=LLMConfig.default_config("gpt-4"),
                    embedding_config=EmbeddingConfig.default_config(
                        "text-embedding-ada-002"
                    ),
                    tool_ids=self.processor_state.metadata_.get(
                        "memory_processing_tools", []
                    ),
                    include_base_tools=False,
                )

                self.background_processor.memory.update_block_value(
                    label="current_interaction_state", value=current_dialog
                )

                session.send_message(
                    agent_id=self.background_processor.id,
                    message="Process and optimize memory structure",
                    role="user",
                )

                session.delete_agent(agent_id=self.background_processor.id)
                self.background_processor = None

            with ThreadPoolExecutor(max_workers=1) as task_executor:
                task_executor.submit(establish_background_processor)

        return interaction_stats
