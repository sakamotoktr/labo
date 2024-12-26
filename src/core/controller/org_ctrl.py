from typing import List, Optional


class EntityHandler:
    """Handles all business processes related to organization entities."""

    DEFAULT_ENTITY_ID = "entity-00000000-0000-4000-8000-000000000000"
    DEFAULT_ENTITY_NAME = "primary_entity"

    def __init__(self):
        self.session_factory = db_context

    @enforce_types
    def retrieve_default_entity(self) -> SchemaOrganization:
        """Retrieve the default organization entity."""
        return self.retrieve_entity_by_id(self.DEFAULT_ENTITY_ID)

    @enforce_types
    def retrieve_entity_by_id(self, entity_id: str) -> Optional[SchemaOrganization]:
        """Retrieve an entity by its unique identifier."""
        with self.session_factory() as session:
            entity = EntityModel.read(db_session=session, identifier=entity_id)
            return entity.to_pydantic()

    @enforce_types
    def initiate_entity(self, schema_entity: SchemaOrganization) -> SchemaOrganization:
        """Initialize and save a new organization entity."""
        try:
            existing_entity = self.retrieve_entity_by_id(schema_entity.id)
            return existing_entity
        except NoResultFound:
            return self._initialize_entity(schema_entity=schema_entity)

    @enforce_types
    def _initialize_entity(
        self, schema_entity: SchemaOrganization
    ) -> SchemaOrganization:
        with self.session_factory() as session:
            entity = EntityModel(**schema_entity.model_dump())
            entity.create(session)
            return entity.to_pydantic()

    @enforce_types
    def initialize_default_entity(self) -> SchemaOrganization:
        """Create and return the default organization entity."""
        return self.initiate_entity(
            SchemaOrganization(name=self.DEFAULT_ENTITY_NAME, id=self.DEFAULT_ENTITY_ID)
        )

    @enforce_types
    def modify_entity_name(
        self, entity_id: str, new_name: Optional[str] = None
    ) -> SchemaOrganization:
        """Change the name of an existing entity using its ID."""
        with self.session_factory() as session:
            entity = EntityModel.read(db_session=session, identifier=entity_id)
            if new_name:
                entity.name = new_name
            entity.update(session)
            return entity.to_pydantic()

    @enforce_types
    def remove_entity_by_id(self, entity_id: str):
        """Delete an entity by performing a permanent deletion."""
        with self.session_factory() as session:
            entity = EntityModel.read(db_session=session, identifier=entity_id)
            entity.hard_delete(session)

    @enforce_types
    def fetch_entities(
        self, start_cursor: Optional[str] = None, max_count: Optional[int] = 50
    ) -> List[SchemaOrganization]:
        """Fetch a list of entities with pagination support."""
        with self.session_factory() as session:
            entities = EntityModel.list(
                db_session=session, cursor=start_cursor, limit=max_count
            )
            return [entity.to_pydantic() for entity in entities]
