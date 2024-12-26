from typing import List, Optional


class ResourceHandler:
    """Handler class to manage operations related to Resources."""

    def __init__(self):
        self.db_session_factory = db_context

    @enforce_types
    def initiate_resource(
        self, resource: PydanticResource, user: PydanticUser
    ) -> PydanticResource:
        """Create a new resource based on the PydanticResource schema."""
        db_resource = self.fetch_resource_by_identifier(resource.id, user=user)
        if db_resource:
            return db_resource
        else:
            with self.db_session_factory() as session:
                resource.organization_id = user.organization_id
                resource = ResourceModel(**resource.model_dump(exclude_none=True))
                resource.create(session, user=user)
            return resource.to_pydantic()

    @enforce_types
    def modify_resource(
        self,
        resource_identifier: str,
        resource_update: ResourceUpdate,
        user: PydanticUser,
    ) -> PydanticResource:
        """Update a resource using the given ResourceUpdate object."""
        with self.db_session_factory() as session:
            resource = ResourceModel.read(
                db_session=session, identifier=resource_identifier, user=user
            )

            # Extract update fields
            update_info = resource_update.model_dump(
                exclude_unset=True, exclude_none=True
            )
            # Remove redundant update fields
            update_info = {
                key: value
                for key, value in update_info.items()
                if getattr(resource, key) != value
            }

            if update_info:
                for key, value in update_info.items():
                    setattr(resource, key, value)
                resource.update(db_session=session, user=user)
            else:
                printd(
                    f"`modify_resource` invoked by user_id={user.id}, organization_id={user.organization_id}, name={resource.name}, but no updates found."
                )

            return resource.to_pydantic()

    @enforce_types
    def eliminate_resource(
        self, resource_identifier: str, user: PydanticUser
    ) -> PydanticResource:
        """Remove a resource using its identifier."""
        with self.db_session_factory() as session:
            resource = ResourceModel.read(
                db_session=session, identifier=resource_identifier
            )
            resource.hard_delete(db_session=session, user=user)
            return resource.to_pydantic()

    @enforce_types
    def retrieve_resources(
        self,
        user: PydanticUser,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        **kwargs,
    ) -> List[PydanticResource]:
        """Fetch all resources with optional pagination."""
        with self.db_session_factory() as session:
            resources = ResourceModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=user.organization_id,
                **kwargs,
            )
            return [resource.to_pydantic() for resource in resources]

    @enforce_types
    def fetch_associated_agents(
        self, resource_identifier: str, user: Optional[PydanticUser] = None
    ) -> List[PydanticAgentState]:
        """
        Fetch all agents associated with the given resource identifier.

        Args:
            resource_identifier: Unique ID of the resource to find associated agents
            user: User performing the action (optional)

        Returns:
            List[PydanticAgentState]: List of agents associated with this resource
        """
        with self.db_session_factory() as session:
            resource = ResourceModel.read(
                db_session=session, identifier=resource_identifier, user=user
            )
            return [agent.to_pydantic() for agent in resource.agents]

    @enforce_types
    def fetch_resource_by_identifier(
        self, resource_identifier: str, user: Optional[PydanticUser] = None
    ) -> Optional[PydanticResource]:
        """Retrieve a resource using its identifier."""
        with self.db_session_factory() as session:
            try:
                resource = ResourceModel.read(
                    db_session=session, identifier=resource_identifier, user=user
                )
                return resource.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def fetch_resource_by_title(
        self, resource_title: str, user: PydanticUser
    ) -> Optional[PydanticResource]:
        """Retrieve a resource using its title."""
        with self.db_session_factory() as session:
            resources = ResourceModel.list(
                db_session=session,
                title=resource_title,
                organization_id=user.organization_id,
                limit=1,
            )
            if not resources:
                return None
            else:
                return resources[0].to_pydantic()

    @enforce_types
    def create_document(
        self, document_metadata: PydanticFileMetadata, user: PydanticUser
    ) -> PydanticFileMetadata:
        """Create a new document based on the PydanticFileMetadata schema."""
        db_document = self.fetch_document_by_identifier(document_metadata.id, user=user)
        if db_document:
            return db_document
        else:
            with self.db_session_factory() as session:
                document_metadata.organization_id = user.organization_id
                document_metadata = FileMetadataModel(
                    **document_metadata.model_dump(exclude_none=True)
                )
                document_metadata.create(session, user=user)
            return document_metadata.to_pydantic()

    @enforce_types
    def fetch_document_by_identifier(
        self, document_identifier: str, user: Optional[PydanticUser] = None
    ) -> Optional[PydanticFileMetadata]:
        """Retrieve a document by its identifier."""
        with self.db_session_factory() as session:
            try:
                document = FileMetadataModel.read(
                    db_session=session, identifier=document_identifier, user=user
                )
                return document.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def list_documents(
        self,
        resource_identifier: str,
        user: PydanticUser,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticFileMetadata]:
        """List all documents with optional pagination."""
        with self.db_session_factory() as session:
            documents = FileMetadataModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=user.organization_id,
                resource_id=resource_identifier,
            )
            return [document.to_pydantic() for document in documents]

    @enforce_types
    def remove_document(
        self, document_identifier: str, user: PydanticUser
    ) -> PydanticFileMetadata:
        """Remove a document using its identifier."""
        with self.db_session_factory() as session:
            document = FileMetadataModel.read(
                db_session=session, identifier=document_identifier
            )
            document.hard_delete(db_session=session, user=user)
            return document.to_pydantic()
