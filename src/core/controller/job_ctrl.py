from typing import List, Optional


class TaskController:
    """Handles processing and management of operational tasks."""

    def __init__(self):
        self._conn_factory = db_context

    @enforce_types
    def fetch_records(
        self,
        auth_entity: AuthEntity,
        page_token: Optional[str] = None,
        batch_size: Optional[int] = 50,
        filters: Optional[List[JobStatus]] = None,
    ) -> List[DataModel]:
        """Extract multiple records based on specified criteria."""
        ctx = self._conn_factory()
        with ctx as persistence:
            query_params = {"user_id": auth_entity.id}

            if filters:
                query_params["status"] = filters

            results = DbRecord.list(
                db_session=persistence,
                cursor=page_token,
                limit=batch_size,
                **query_params,
            )
            return [item.to_pydantic() for item in results]

    @enforce_types
    def remove_entry(self, entry_id: str, operator: AuthEntity) -> DataModel:
        """Permanently eliminate specified record."""
        ctx = self._conn_factory()
        with ctx as persistence:
            entry = DbRecord.read(db_session=persistence, identifier=entry_id)
            entry.hard_delete(db_session=persistence)
            return entry.to_pydantic()

    @enforce_types
    def retrieve_entry(self, entry_id: str, operator: AuthEntity) -> DataModel:
        """Extract single record details."""
        ctx = self._conn_factory()
        with ctx as persistence:
            result = DbRecord.read(db_session=persistence, identifier=entry_id)
            return result.to_pydantic()

    @enforce_types
    def persist_entry(self, payload: DataModel, operator: AuthEntity) -> DataModel:
        """Store new operational record."""
        ctx = self._conn_factory()
        with ctx as persistence:
            payload.user_id = operator.id
            raw_data = payload.model_dump()
            record = DbRecord(**raw_data)
            record.create(persistence, actor=operator)
        return record.to_pydantic()

    @enforce_types
    def modify_entry(
        self, entry_id: str, modifications: JobUpdate, operator: AuthEntity
    ) -> DataModel:
        """Apply modifications to existing record."""
        ctx = self._conn_factory()
        with ctx as persistence:
            target = DbRecord.read(db_session=persistence, identifier=entry_id)
            delta = modifications.model_dump(exclude_unset=True, exclude_none=True)

            if delta.get("status") == JobStatus.completed and not target.completed_at:
                target.completed_at = get_utc_time()

            for attr, val in delta.items():
                setattr(target, attr, val)

            return target.update(db_session=persistence)
