from datetime import datetime
from typing import Dict, List, Optional


class MsgHandler:
    """Handles core operations related to messaging functionality."""

    def __init__(self):
        self.db_session_creator = db_context

    @enforce_types
    def retrieve_msg_by_id(self, msg_id: str, user: UserSchema) -> Optional[MsgSchema]:
        """Retrieve a specific message based on its identifier."""
        with self.db_session_creator() as db_session:
            try:
                msg = MsgModel.read(
                    db_session=db_session, identifier=msg_id, actor=user
                )
                return msg.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def retrieve_msgs_by_ids(
        self, msg_ids: List[str], user: UserSchema
    ) -> List[MsgSchema]:
        """Retrieve messages by their identifiers, maintaining the requested order."""
        with self.db_session_creator() as db_session:
            messages = MsgModel.list(
                db_session=db_session,
                id=msg_ids,
                organization_id=user.organization_id,
                limit=len(msg_ids),
            )

            if len(messages) != len(msg_ids):
                missing_ids = set(msg_ids) - set([msg.id for msg in messages])
                raise NoResultFound(f"Missing ids={missing_ids}")

            # Return the results in the original order of msg_ids
            message_map = {msg.id: msg.to_pydantic() for msg in messages}
            return [message_map[msg_id] for msg_id in msg_ids]

    @enforce_types
    def create_new_msg(self, msg_data: MsgSchema, user: UserSchema) -> MsgSchema:
        """Create and persist a new message."""
        with self.db_session_creator() as db_session:
            msg_data.organization_id = user.organization_id
            msg_payload = msg_data.model_dump()
            new_msg = MsgModel(**msg_payload)
            new_msg.create(db_session, actor=user)
            return new_msg.to_pydantic()

    @enforce_types
    def bulk_create_msgs(
        self, msg_data_list: List[MsgSchema], user: UserSchema
    ) -> List[MsgSchema]:
        """Create multiple messages in one operation."""
        return [self.create_new_msg(msg_data, user) for msg_data in msg_data_list]

    @enforce_types
    def update_msg_by_id(
        self, msg_id: str, update_data: MsgUpdate, user: UserSchema
    ) -> MsgSchema:
        """
        Update an existing message with new data from the provided update object.
        """
        with self.db_session_creator() as db_session:
            msg = MsgModel.read(db_session=db_session, identifier=msg_id, actor=user)

            # Perform checks for safety based on message role
            if update_data.tool_calls and msg.role != MsgRole.assistant:
                raise ValueError(
                    f"Tool calls can only be added to assistant messages. Msg {msg_id} has role {msg.role}."
                )
            if update_data.tool_call_id and msg.role != MsgRole.tool:
                raise ValueError(
                    f"Tool call IDs can only be added to tool messages. Msg {msg_id} has role {msg.role}."
                )

            # Get update data, excluding unset and None fields
            update_fields = update_data.model_dump(
                exclude_unset=True, exclude_none=True
            )
            update_fields = {
                key: value
                for key, value in update_fields.items()
                if getattr(msg, key) != value
            }

            for field, new_value in update_fields.items():
                setattr(msg, field, new_value)
            msg.update(db_session=db_session, actor=user)

            return msg.to_pydantic()

    @enforce_types
    def delete_msg_by_id(self, msg_id: str, user: UserSchema) -> bool:
        """Delete a message from the system by its identifier."""
        with self.db_session_creator() as db_session:
            try:
                msg = MsgModel.read(
                    db_session=db_session, identifier=msg_id, actor=user
                )
                msg.hard_delete(db_session, actor=user)
            except NoResultFound:
                raise ValueError(f"Message with id {msg_id} not found.")

    @enforce_types
    def count_msgs(
        self,
        user: UserSchema,
        role: Optional[MsgRole] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Count the total number of messages with optional filters."""
        with self.db_session_creator() as db_session:
            return MsgModel.size(
                db_session=db_session, actor=user, role=role, agent_id=agent_id
            )

    @enforce_types
    def list_msgs_by_user_for_agent(
        self,
        agent_id: str,
        user: Optional[UserSchema] = None,
        cursor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        search_text: Optional[str] = None,
        is_ascending: bool = True,
    ) -> List[MsgSchema]:
        """List user-specific messages for a given agent with various filtering options."""
        filters = filters or {}
        filters["role"] = "user"

        return self.list_msgs_for_agent(
            agent_id=agent_id,
            user=user,
            cursor=cursor,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            filters=filters,
            search_text=search_text,
            is_ascending=is_ascending,
        )

    @enforce_types
    def list_msgs_for_agent(
        self,
        agent_id: str,
        user: Optional[UserSchema] = None,
        cursor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = 50,
        filters: Optional[Dict] = None,
        search_text: Optional[str] = None,
        is_ascending: bool = True,
    ) -> List[MsgSchema]:
        """List messages associated with an agent, applying filters and pagination."""
        with self.db_session_creator() as db_session:
            base_filters = {"agent_id": agent_id}
            if user:
                base_filters["organization_id"] = user.organization_id
            if filters:
                base_filters.update(filters)

            messages = MsgModel.list(
                db_session=db_session,
                cursor=cursor,
                start_date=start_time,
                end_date=end_time,
                limit=limit,
                query_text=search_text,
                ascending=is_ascending,
                **base_filters,
            )

            return [msg.to_pydantic() for msg in messages]
