from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import Select, func, literal, select, union_all


custom_logger = setup_custom_logger(__name__)


class EntityProcessor:
    def __init__(self):
        self.session_factory = storage_context
        self.storage_ctrl = StorageController()
        self.func_handler = FunctionalityHandler()
        self.origin_ctrl = OriginController()
        self.comm_handler = CommunicationHandler()

    @verify_input_types
    def initialize_entity(
        self, entity_init: InitEntity, executor: PydanticUser
    ) -> PydanticAgentState:
        instructions = generate_instruction_set(
            category=entity_init.agent_type, custom_instructions=entity_init.system
        )

        if not entity_init.llm_config or not entity_init.embedding_config:
            raise ValueError("processor and vector configurations required")

        if entity_init.tool_rules:
            validate_output_format(
                processor=entity_init.llm_config.model, rules=entity_init.tool_rules
            )

        storage_ids = list(entity_init.block_ids or [])
        for new_storage in entity_init.memory_blocks:
            storage_unit = self.storage_ctrl.create_or_modify_storage(
                PydanticBlock(**new_storage.model_dump()), executor=executor
            )
            storage_ids.append(storage_unit.id)

        functionality_names = []
        if entity_init.include_base_tools:
            functionality_names.extend(DEFAULT_TOOLSET + DEFAULT_BASE_FUNCTIONALITY)
        if entity_init.tools:
            functionality_names.extend(entity_init.tools)
        functionality_names = list(set(functionality_names))

        func_ids = entity_init.tool_ids or []
        for func_name in functionality_names:
            functionality = self.func_handler.retrieve_functionality_by_name(
                func_name=func_name, executor=executor
            )
            if functionality:
                func_ids.append(functionality.id)
        func_ids = list(set(func_ids))

        # Create the entity
        entity_status = self._create_entity(
            name=entity_init.name,
            instructions=instructions,
            category=entity_init.agent_type,
            processor_config=entity_init.llm_config,
            vector_config=entity_init.embedding_config,
            storage_ids=storage_ids,
            func_ids=func_ids,
            origin_ids=entity_init.source_ids or [],
            markers=entity_init.tags or [],
            description=entity_init.description,
            metadata=entity_init.metadata_,
            rules=entity_init.tool_rules,
            executor=executor,
        )

        # Generate a sequence of initial messages to put in the buffer
        init_messages = setup_initial_sequence(
            entity_status=entity_status,
            memory_edit_timestamp=get_current_time(),
            include_initial_boot_message=True,
        )

        if entity_init.initial_message_sequence is not None:
            # We always need the system prompt up front
            system_message_obj = PydanticMessage.dict_to_message(
                agent_id=entity_status.id,
                user_id=entity_status.created_by_id,
                model=entity_status.llm_config.model,
                openai_message_dict=init_messages[0],
            )
            # Don't use anything else in the pregen sequence, instead use the provided sequence
            init_messages = [system_message_obj]
            init_messages.extend(
                prepare_startup_sequence(
                    entity_status.id,
                    entity_init.initial_message_sequence,
                    entity_status.llm_config.model,
                    executor,
                )
            )
        else:
            init_messages = [
                PydanticMessage.dict_to_message(
                    agent_id=entity_status.id,
                    user_id=entity_status.created_by_id,
                    model=entity_status.llm_config.model,
                    openai_message_dict=msg,
                )
                for msg in init_messages
            ]

        return self.append_to_in_context_messages(
            init_messages, agent_id=entity_status.id, executor=executor
        )

    @verify_input_types
    def _create_entity(
        self,
        executor: PydanticUser,
        name: str,
        instructions: str,
        category: EntityCategory,
        processor_config: ProcessorConfig,
        vector_config: VectorConfig,
        storage_ids: List[str],
        func_ids: List[str],
        origin_ids: List[str],
        markers: List[str],
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        rules: Optional[List[PydanticToolRule]] = None,
    ) -> PydanticAgentState:
        """Create a new entity."""
        with self.session_factory() as session:
            # Prepare the entity data
            data = {
                "name": name,
                "instructions": instructions,
                "category": category,
                "processor_config": processor_config,
                "vector_config": vector_config,
                "organization_id": executor.organization_id,
                "description": description,
                "metadata": metadata,
                "rules": rules,
            }

            # Create the new entity using SqlalchemyBase.create
            new_entity = AgentModel(**data)
            _handle_connection(
                session, new_entity, "tools", ToolModel, func_ids, replace=True
            )
            _handle_connection(
                session, new_entity, "origins", SourceModel, origin_ids, replace=True
            )
            _handle_connection(
                session,
                new_entity,
                "core_memory",
                BlockModel,
                storage_ids,
                replace=True,
            )
            _handle_markers(new_entity, markers, replace=True)
            new_entity.create(session, executor=executor)

            # Convert to PydanticAgentState and return
            return new_entity.to_pydantic()

    @verify_input_types
    def modify_entity(
        self, entity_id: str, entity_update: ModifyEntity, executor: PydanticUser
    ) -> PydanticAgentState:
        entity_status = self._modify_entity(
            entity_id=entity_id, entity_update=entity_update, executor=executor
        )

        # Rebuild the system prompt if it's different
        if entity_update.system and entity_update.system != entity_status.system:
            entity_status = self.rebuild_system_prompt(
                entity_id=entity_status.id,
                executor=executor,
                force=True,
                update_timestamp=False,
            )

        return entity_status

    @verify_input_types
    def _modify_entity(
        self, entity_id: str, entity_update: ModifyEntity, executor: PydanticUser
    ) -> PydanticAgentState:
        """
        Update an existing entity.

        Args:
            entity_id: The ID of the entity to update.
            entity_update: ModifyEntity object containing the updated fields.
            executor: User performing the action.

        Returns:
            PydanticAgentState: The updated entity as a Pydantic model.
        """
        with self.session_factory() as session:
            # Retrieve the existing entity
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # Update scalar fields directly
            scalar_fields = {
                "name",
                "instructions",
                "processor_config",
                "vector_config",
                "message_ids",
                "rules",
                "description",
                "metadata",
            }
            for field in scalar_fields:
                value = getattr(entity_update, field, None)
                if value is not None:
                    setattr(entity, field, value)

            # Update relationships using _handle_connection and _handle_markers
            if entity_update.tool_ids is not None:
                _handle_connection(
                    session,
                    entity,
                    "tools",
                    ToolModel,
                    entity_update.tool_ids,
                    replace=True,
                )
            if entity_update.source_ids is not None:
                _handle_connection(
                    session,
                    entity,
                    "origins",
                    SourceModel,
                    entity_update.source_ids,
                    replace=True,
                )
            if entity_update.block_ids is not None:
                _handle_connection(
                    session,
                    entity,
                    "core_memory",
                    BlockModel,
                    entity_update.block_ids,
                    replace=True,
                )
            if entity_update.tags is not None:
                _handle_markers(entity, entity_update.tags, replace=True)

            # Commit and refresh the entity
            entity.update(session, executor=executor)

            # Convert to PydanticAgentState and return
            return entity.to_pydantic()

    @verify_input_types
    def list_entities(
        self,
        executor: PydanticUser,
        markers: Optional[List[str]] = None,
        match_all_markers: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        **kwargs,
    ) -> List[PydanticAgentState]:
        """
        List entities that have the specified markers.
        """
        with self.session_factory() as session:
            entities = AgentModel.list(
                db_session=session,
                tags=markers,
                match_all_tags=match_all_markers,
                cursor=cursor,
                limit=limit,
                organization_id=executor.organization_id if executor else None,
                **kwargs,
            )

            return [entity.to_pydantic() for entity in entities]

    @verify_input_types
    def get_entity_by_id(
        self, entity_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """Fetch an entity by its ID."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            return entity.to_pydantic()

    @verify_input_types
    def get_entity_by_name(
        self, entity_name: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """Fetch an entity by its ID."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, name=entity_name, executor=executor
            )
            return entity.to_pydantic()

    @verify_input_types
    def delete_entity(
        self, entity_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """
        Deletes an entity and its associated relationships.
        Ensures proper permission checks and cascades where applicable.

        Args:
            entity_id: ID of the entity to be deleted.
            executor: User performing the action.

        Returns:
            PydanticAgentState: The deleted entity state
        """
        with self.session_factory() as session:
            # Retrieve the entity
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            entity_status = entity.to_pydantic()
            entity.hard_delete(session)
            return entity_status

    @verify_input_types
    def get_in_context_messages(
        self, entity_id: str, executor: PydanticUser
    ) -> List[PydanticMessage]:
        message_ids = self.get_entity_by_id(
            entity_id=entity_id, executor=executor
        ).message_ids
        return self.comm_handler.get_messages_by_ids(
            message_ids=message_ids, executor=executor
        )

    @verify_input_types
    def get_system_message(
        self, entity_id: str, executor: PydanticUser
    ) -> PydanticMessage:
        message_ids = self.get_entity_by_id(
            entity_id=entity_id, executor=executor
        ).message_ids
        return self.comm_handler.get_message_by_id(
            message_id=message_ids[0], executor=executor
        )

    @verify_input_types
    def rebuild_system_prompt(
        self, entity_id: str, executor: PydanticUser, force=False, update_timestamp=True
    ) -> PydanticAgentState:
        """Rebuilds the system message with the latest memory object and any shared memory block updates

        Updates to core memory blocks should trigger a "rebuild", which itself will create a new message object

        Updates to the memory header should *not* trigger a rebuild, since that will simply flood recall storage with excess messages
        """
        entity_status = self.get_entity_by_id(entity_id=entity_id, executor=executor)

        curr_system_message = self.get_system_message(
            entity_id=entity_id, executor=executor
        )  # this is the system + memory bank, not just the system prompt
        curr_system_message_openai = curr_system_message.to_openai_dict()

        # note: we only update the system prompt if the core memory is changed
        # this means that the archival/recall memory statistics may be someout out of date
        curr_memory_str = entity_status.memory.compile()
        if curr_memory_str in curr_system_message_openai["content"] and not force:
            # NOTE: could this cause issues if a block is removed? (substring match would still work)
            custom_logger.debug(
                f"Memory hasn't changed for entity id={entity_id} and executor=({executor.id}, {executor.name}), skipping system prompt rebuild"
            )
            return entity_status

        # If the memory didn't update, we probably don't want to update the timestamp inside
        # For example, if we're doing a system prompt swap, this should probably be False
        if update_timestamp:
            memory_edit_timestamp = get_current_time()
        else:
            # NOTE: a bit of a hack - we pull the timestamp from the message created_by
            memory_edit_timestamp = curr_system_message.created_at

        # update memory (TODO: potentially update recall/archival stats separately)
        new_system_message_str = build_instruction_set(
            system_prompt=entity_status.system,
            in_context_memory=entity_status.memory,
            in_context_memory_last_edit=memory_edit_timestamp,
        )

        diff = text_comparison(
            curr_system_message_openai["content"], new_system_message_str
        )
        if len(diff) > 0:  # there was a diff
            custom_logger.info(f"Rebuilding system with new memory...\nDiff:\n{diff}")

            # Swap the system message out (only if there is a diff)
            message = PydanticMessage.dict_to_message(
                agent_id=entity_id,
                user_id=executor.id,
                model=entity_status.llm_config.model,
                openai_message_dict={
                    "role": "system",
                    "content": new_system_message_str,
                },
            )
            message = self.comm_handler.create_message(message, executor=executor)
            message_ids = [message.id] + entity_status.message_ids[
                1:
            ]  # swap index 0 (system)
            return self.set_in_context_messages(
                entity_id=entity_id, message_ids=message_ids, executor=executor
            )
        else:
            return entity_status

    @verify_input_types
    def set_in_context_messages(
        self, entity_id: str, message_ids: List[str], executor: PydanticUser
    ) -> PydanticAgentState:
        return self.modify_entity(
            entity_id=entity_id,
            entity_update=ModifyEntity(message_ids=message_ids),
            executor=executor,
        )

    @verify_input_types
    def trim_older_in_context_messages(
        self, num: int, entity_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        message_ids = self.get_entity_by_id(
            entity_id=entity_id, executor=executor
        ).message_ids
        new_messages = [message_ids[0]] + message_ids[num:]  # 0 is system message
        return self.set_in_context_messages(
            entity_id=entity_id, message_ids=new_messages, executor=executor
        )

    @verify_input_types
    def prepend_to_in_context_messages(
        self, messages: List[PydanticMessage], entity_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        message_ids = self.get_entity_by_id(
            entity_id=entity_id, executor=executor
        ).message_ids
        new_messages = self.comm_handler.create_many_messages(
            messages, executor=executor
        )
        message_ids = [message_ids[0]] + [m.id for m in new_messages] + message_ids[1:]
        return self.set_in_context_messages(
            entity_id=entity_id, message_ids=message_ids, executor=executor
        )

    @verify_input_types
    def append_to_in_context_messages(
        self, messages: List[PydanticMessage], entity_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        messages = self.comm_handler.create_many_messages(messages, executor=executor)
        message_ids = (
            self.get_entity_by_id(entity_id=entity_id, executor=executor).message_ids
            or []
        )
        message_ids += [m.id for m in messages]
        return self.set_in_context_messages(
            entity_id=entity_id, message_ids=message_ids, executor=executor
        )

    @verify_input_types
    def attach_origin(
        self, entity_id: str, origin_id: str, executor: PydanticUser
    ) -> None:
        """
        Attaches a origin to an entity.

        Args:
            entity_id: ID of the entity to attach the origin to
            origin_id: ID of the origin to attach
            executor: User performing the action

        Raises:
            ValueError: If either entity or origin doesn't exist
            IntegrityError: If the origin is already attached to the entity
        """
        with self.session_factory() as session:
            # Verify both entity and origin exist and user has permission to access them
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # The _handle_connection helper already handles duplicate checking via unique constraint
            _handle_connection(
                session=session,
                agent=entity,
                relationship_name="origins",
                model_class=SourceModel,
                item_ids=[origin_id],
                allow_partial=False,
                replace=False,  # Extend existing origins rather than replace
            )

            # Commit the changes
            entity.update(session, executor=executor)

    @verify_input_types
    def list_attached_origins(
        self, entity_id: str, executor: PydanticUser
    ) -> List[PydanticSource]:
        """
        Lists all origins attached to an entity.

        Args:
            entity_id: ID of the entity to list origins for
            executor: User performing the action

        Returns:
            List[str]: List of origin IDs attached to the entity
        """
        with self.session_factory() as session:
            # Verify entity exists and user has permission to access it
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # Use the lazy-loaded relationship to get origins
            return [origin.to_pydantic() for origin in entity.origins]

    @verify_input_types
    def detach_origin(
        self, entity_id: str, origin_id: str, executor: PydanticUser
    ) -> None:
        """
        Detaches a origin from an entity.

        Args:
            entity_id: ID of the entity to detach the origin from
            origin_id: ID of the origin to detach
            executor: User performing the action
        """
        with self.session_factory() as session:
            # Verify entity exists and user has permission to access it
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # Remove the origin from the relationship
            entity.origins = [s for s in entity.origins if s.id != origin_id]

            # Commit the changes
            entity.update(session, executor=executor)

    @verify_input_types
    def get_storage_with_marker(
        self,
        entity_id: str,
        storage_marker: str,
        executor: PydanticUser,
    ) -> PydanticBlock:
        """Gets a storage attached to an entity by its marker."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            for storage in entity.core_memory:
                if storage.label == storage_marker:
                    return storage.to_pydantic()
            raise ResourceNotFound(
                f"No storage with marker '{storage_marker}' found for entity '{entity_id}'"
            )

    @verify_input_types
    def update_storage_with_marker(
        self,
        entity_id: str,
        storage_marker: str,
        new_storage_id: str,
        executor: PydanticUser,
    ) -> PydanticAgentState:
        """Updates which storage is assigned to a specific marker for an entity."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            new_storage = BlockModel.read(
                db_session=session, identifier=new_storage_id, executor=executor
            )

            if new_storage.label != storage_marker:
                raise ValueError(
                    f"New storage marker '{new_storage.label}' doesn't match required marker '{storage_marker}'"
                )

            # Remove old storage with this marker if it exists
            entity.core_memory = [
                b for b in entity.core_memory if b.label != storage_marker
            ]

            # Add new storage
            entity.core_memory.append(new_storage)
            entity.update(session, executor=executor)
            return entity.to_pydantic()

    @verify_input_types
    def attach_storage(
        self, entity_id: str, storage_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """Attaches a storage to an entity."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            storage = BlockModel.read(
                db_session=session, identifier=storage_id, executor=executor
            )

            entity.core_memory.append(storage)
            entity.update(session, executor=executor)
            return entity.to_pydantic()

    @verify_input_types
    def detach_storage(
        self,
        entity_id: str,
        storage_id: str,
        executor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a storage from an entity."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            original_length = len(entity.core_memory)

            entity.core_memory = [b for b in entity.core_memory if b.id != storage_id]

            if len(entity.core_memory) == original_length:
                raise ResourceNotFound(
                    f"No storage with id '{storage_id}' found for entity '{entity_id}' with executor id: '{executor.id}'"
                )

            entity.update(session, executor=executor)
            return entity.to_pydantic()

    @verify_input_types
    def detach_storage_with_marker(
        self,
        entity_id: str,
        storage_marker: str,
        executor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a storage with the specified marker from an entity."""
        with self.session_factory() as session:
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )
            original_length = len(entity.core_memory)

            entity.core_memory = [
                b for b in entity.core_memory if b.label != storage_marker
            ]

            if len(entity.core_memory) == original_length:
                raise ResourceNotFound(
                    f"No storage with marker '{storage_marker}' found for entity '{entity_id}' with executor id: '{executor.id}'"
                )

            entity.update(session, executor=executor)
            return entity.to_pydantic()

    def _build_segment_query(
        self,
        executor: PydanticUser,
        entity_id: Optional[str] = None,
        file_id: Optional[str] = None,
        query_text: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
        origin_id: Optional[str] = None,
        embed_query: bool = False,
        ascending: bool = True,
        vector_config: Optional[VectorConfig] = None,
        entity_only: bool = False,
    ) -> Select:
        """Helper function to build the base segment query with all filters applied.

        Returns the query before any limit or count operations are applied.
        """
        embedded_text = None
        if embed_query:
            assert (
                vector_config is not None
            ), "vector_config must be specified for vector search"
            assert (
                query_text is not None
            ), "query_text must be specified for vector search"
            embedded_text = vector_processor(vector_config).get_text_embedding(
                query_text
            )
            embedded_text = np.array(embedded_text)
            embedded_text = np.pad(
                embedded_text,
                (0, VECTOR_MAX_DIM - embedded_text.shape[0]),
                mode="constant",
            ).tolist()

        with self.session_factory() as session:
            # Start with base query for origin segments
            origin_segments = None
            if not entity_only:  # Include origin segments
                if entity_id is not None:
                    origin_segments = (
                        select(OriginSegment, literal(None).label("entity_id"))
                        .join(
                            OriginsEntities,
                            OriginsEntities.source_id == OriginSegment.source_id,
                        )
                        .where(OriginsEntities.agent_id == entity_id)
                        .where(
                            OriginSegment.organization_id == executor.organization_id
                        )
                    )
                else:
                    origin_segments = select(
                        OriginSegment, literal(None).label("entity_id")
                    ).where(OriginSegment.organization_id == executor.organization_id)

                if origin_id:
                    origin_segments = origin_segments.where(
                        OriginSegment.source_id == origin_id
                    )
                if file_id:
                    origin_segments = origin_segments.where(
                        OriginSegment.file_id == file_id
                    )

            # Add entity segments query
            entity_segments = None
            if entity_id is not None:
                entity_segments = (
                    select(
                        EntitySegment.id,
                        EntitySegment.text,
                        EntitySegment.vector_config,
                        EntitySegment.metadata,
                        EntitySegment.embedding,
                        EntitySegment.created_at,
                        EntitySegment.updated_at,
                        EntitySegment.is_deleted,
                        EntitySegment._created_by_id,
                        EntitySegment._last_updated_by_id,
                        EntitySegment.organization_id,
                        literal(None).label("file_id"),
                        literal(None).label("source_id"),
                        EntitySegment.agent_id,
                    )
                    .where(EntitySegment.agent_id == entity_id)
                    .where(EntitySegment.organization_id == executor.organization_id)
                )

            # Combine queries
            if origin_segments is not None and entity_segments is not None:
                combined_query = union_all(origin_segments, entity_segments).cte(
                    "combined_segments"
                )
            elif entity_segments is not None:
                combined_query = entity_segments.cte("combined_segments")
            elif origin_segments is not None:
                combined_query = origin_segments.cte("combined_segments")
            else:
                raise ValueError("No segments found")

            # Build main query from combined CTE
            main_query = select(combined_query)

            # Apply filters
            if start_date:
                main_query = main_query.where(combined_query.c.created_at >= start_date)
            if end_date:
                main_query = main_query.where(combined_query.c.created_at <= end_date)
            if origin_id:
                main_query = main_query.where(combined_query.c.source_id == origin_id)
            if file_id:
                main_query = main_query.where(combined_query.c.file_id == file_id)

            # Vector search
            if embedded_text:
                if configuration.labo_pg_uri_no_default:
                    # PostgreSQL with pgvector
                    main_query = main_query.order_by(
                        combined_query.c.embedding.cosine_distance(embedded_text).asc()
                    )
                else:
                    # SQLite with custom vector type
                    query_embedding_binary = process_vector(embedded_text)
                    if ascending:
                        main_query = main_query.order_by(
                            func.cosine_distance(
                                combined_query.c.embedding, query_embedding_binary
                            ).asc(),
                            combined_query.c.created_at.asc(),
                            combined_query.c.id.asc(),
                        )
                    else:
                        main_query = main_query.order_by(
                            func.cosine_distance(
                                combined_query.c.embedding, query_embedding_binary
                            ).asc(),
                            combined_query.c.created_at.desc(),
                            combined_query.c.id.asc(),
                        )
            else:
                if query_text:
                    main_query = main_query.where(
                        func.lower(combined_query.c.text).contains(
                            func.lower(query_text)
                        )
                    )

            # Handle cursor-based pagination
            if cursor:
                cursor_query = (
                    select(combined_query.c.created_at)
                    .where(combined_query.c.id == cursor)
                    .scalar_subquery()
                )

                if ascending:
                    main_query = main_query.where(
                        combined_query.c.created_at > cursor_query
                    )
                else:
                    main_query = main_query.where(
                        combined_query.c.created_at < cursor_query
                    )

            # Add ordering if not already ordered by similarity
            if not embed_query:
                if ascending:
                    main_query = main_query.order_by(
                        combined_query.c.created_at.asc(),
                        combined_query.c.id.asc(),
                    )
                else:
                    main_query = main_query.order_by(
                        combined_query.c.created_at.desc(),
                        combined_query.c.id.asc(),
                    )

            return main_query

    @verify_input_types
    def list_segments(
        self,
        executor: PydanticUser,
        entity_id: Optional[str] = None,
        file_id: Optional[str] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
        origin_id: Optional[str] = None,
        embed_query: bool = False,
        ascending: bool = True,
        vector_config: Optional[VectorConfig] = None,
        entity_only: bool = False,
    ) -> List[PydanticPassage]:
        """Lists all segments attached to an entity."""
        with self.session_factory() as session:
            main_query = self._build_segment_query(
                executor=executor,
                entity_id=entity_id,
                file_id=file_id,
                query_text=query_text,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor,
                origin_id=origin_id,
                embed_query=embed_query,
                ascending=ascending,
                vector_config=vector_config,
                entity_only=entity_only,
            )

            # Add limit
            if limit:
                main_query = main_query.limit(limit)

            # Execute query
            results = list(session.execute(main_query))

            segments = []
            for row in results:
                data = dict(row._mapping)
                if data["agent_id"] is not None:
                    # This is an EntitySegment - remove origin fields
                    data.pop("source_id", None)
                    data.pop("file_id", None)
                    segment = EntitySegment(**data)
                else:
                    # This is a OriginSegment - remove entity field
                    data.pop("agent_id", None)
                    segment = OriginSegment(**data)
                segments.append(segment)

            return [p.to_pydantic() for p in segments]

    @verify_input_types
    def segment_size(
        self,
        executor: PydanticUser,
        entity_id: Optional[str] = None,
        file_id: Optional[str] = None,
        query_text: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
        origin_id: Optional[str] = None,
        embed_query: bool = False,
        ascending: bool = True,
        vector_config: Optional[VectorConfig] = None,
        entity_only: bool = False,
    ) -> int:
        """Returns the count of segments matching the given criteria."""
        with self.session_factory() as session:
            main_query = self._build_segment_query(
                executor=executor,
                entity_id=entity_id,
                file_id=file_id,
                query_text=query_text,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor,
                origin_id=origin_id,
                embed_query=embed_query,
                ascending=ascending,
                vector_config=vector_config,
                entity_only=entity_only,
            )

            # Convert to count query
            count_query = select(func.count()).select_from(main_query.subquery())
            return session.scalar(count_query) or 0

    @verify_input_types
    def attach_functionality(
        self, entity_id: str, func_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """
        Attaches a functionality to an entity.

        Args:
            entity_id: ID of the entity to attach the functionality to.
            func_id: ID of the functionality to attach.
            executor: User performing the action.

        Raises:
            ResourceNotFound: If the entity or functionality is not found.

        Returns:
            PydanticAgentState: The updated entity state.
        """
        with self.session_factory() as session:
            # Verify the entity exists and user has permission to access it
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # Use the _handle_connection helper to attach the functionality
            _handle_connection(
                session=session,
                agent=entity,
                relationship_name="tools",
                model_class=ToolModel,
                item_ids=[func_id],
                allow_partial=False,  # Ensure the functionality exists
                replace=False,  # Extend the existing functionalities
            )

            # Commit and refresh the entity
            entity.update(session, executor=executor)
            return entity.to_pydantic()

    @verify_input_types
    def detach_functionality(
        self, entity_id: str, func_id: str, executor: PydanticUser
    ) -> PydanticAgentState:
        """
        Detaches a functionality from an entity.

        Args:
            entity_id: ID of the entity to detach the functionality from.
            func_id: ID of the functionality to detach.
            executor: User performing the action.

        Raises:
            ResourceNotFound: If the entity or functionality is not found.

        Returns:
            PydanticAgentState: The updated entity state.
        """
        with self.session_factory() as session:
            # Verify the entity exists and user has permission to access it
            entity = AgentModel.read(
                db_session=session, identifier=entity_id, executor=executor
            )

            # Filter out the functionality to be detached
            remaining_functionalities = [
                func for func in entity.tools if func.id != func_id
            ]

            if len(remaining_functionalities) == len(
                entity.tools
            ):  # Functionality ID was not in the relationship
                custom_logger.warning(
                    f"Attempted to remove unattached functionality id={func_id} from entity id={entity_id} by executor={executor}"
                )

            # Update the functionalities relationship
            entity.tools = remaining_functionalities

            # Commit and refresh the entity
            entity.update(session, executor=executor)
            return entity.to_pydantic()
