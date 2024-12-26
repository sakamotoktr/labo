from typing import List, Optional
from datetime import datetime
import numpy as np

from sqlalchemy import select, union_all, literal


class DocumentController:
    """Class responsible for managing document-related operations."""

    def __init__(self):
        self.session_creator = db_context

    @enforce_types
    def fetch_document_by_identifier(
        self, doc_identifier: str, user: PydanticUser
    ) -> Optional[PydanticDocument]:
        """Retrieve a document by its unique identifier."""
        with self.session_creator() as session:
            # Attempt to fetch from primary document storage
            try:
                doc = PrimaryDocument.read(
                    db_session=session, identifier=doc_identifier, user=user
                )
                return doc.to_pydantic()
            except NoResultFound:
                # Fallback to secondary document storage
                try:
                    doc = SecondaryDocument.read(
                        db_session=session, identifier=doc_identifier, user=user
                    )
                    return doc.to_pydantic()
                except NoResultFound:
                    raise NoResultFound(
                        f"Document with id {doc_identifier} not found in database."
                    )

    @enforce_types
    def add_new_document(
        self, pydantic_document: PydanticDocument, user: PydanticUser
    ) -> PydanticDocument:
        """Insert a new document into the appropriate storage based on its metadata."""
        # Common document fields
        doc_data = pydantic_document.model_dump()
        base_fields = {
            "id": doc_data.get("id"),
            "content": doc_data["content"],
            "vector": doc_data["vector"],
            "vector_config": doc_data["vector_config"],
            "organization_id": doc_data["organization_id"],
            "metadata_": doc_data.get("metadata_", {}),
            "is_deleted": doc_data.get("is_deleted", False),
            "created_at": doc_data.get("created_at", datetime.utcnow()),
        }

        if "agent_id" in doc_data and doc_data["agent_id"]:
            assert not doc_data.get(
                "source_id"
            ), "Document cannot have both agent_id and source_id"
            agent_specific_fields = {
                "agent_id": doc_data["agent_id"],
            }
            document = SecondaryDocument(**base_fields, **agent_specific_fields)
        elif "source_id" in doc_data and doc_data["source_id"]:
            assert not doc_data.get(
                "agent_id"
            ), "Document cannot have both agent_id and source_id"
            source_specific_fields = {
                "source_id": doc_data["source_id"],
                "file_id": doc_data.get("file_id"),
            }
            document = PrimaryDocument(**base_fields, **source_specific_fields)
        else:
            raise ValueError("Document must have either agent_id or source_id")

        with self.session_creator() as session:
            document.create(session, user=user)
            return document.to_pydantic()

    @enforce_types
    def add_multiple_documents(
        self, documents: List[PydanticDocument], user: PydanticUser
    ) -> List[PydanticDocument]:
        """Insert a batch of documents."""
        return [self.add_new_document(doc, user) for doc in documents]

    @enforce_types
    def insert_document(
        self,
        agent_status: AgentState,
        agent_identifier: str,
        content: str,
        user: PydanticUser,
    ) -> List[PydanticDocument]:
        """Insert documents into the archival system."""

        chunk_size = agent_status.vector_config.embedding_chunk_size
        vectorizer = vector_model(agent_status.vector_config)

        documents = []

        try:
            # Split content into individual documents
            for chunk in split_and_chunk_text(content, chunk_size):
                vector_rep = vectorizer.get_text_vector(chunk)
                if isinstance(vector_rep, dict):
                    try:
                        vector_rep = vector_rep["data"][0]["vector"]
                    except (KeyError, IndexError):
                        raise TypeError(
                            f"Received an unexpected response from vector generation function, type={type(vector_rep)}, value={vector_rep}"
                        )
                document = self.add_new_document(
                    PydanticDocument(
                        organization_id=user.organization_id,
                        agent_id=agent_identifier,
                        content=chunk,
                        vector=vector_rep,
                        vector_config=agent_status.vector_config,
                    ),
                    user=user,
                )
                documents.append(document)

            return documents

        except Exception as error:
            raise error

    @enforce_types
    def update_document_by_identifier(
        self,
        doc_identifier: str,
        updated_document: PydanticDocument,
        user: PydanticUser,
        **kwargs,
    ) -> Optional[PydanticDocument]:
        """Update an existing document."""
        if not doc_identifier:
            raise ValueError("Document identifier is required.")

        with self.session_creator() as session:
            # Attempt to fetch from primary storage first
            try:
                existing_doc = PrimaryDocument.read(
                    db_session=session,
                    identifier=doc_identifier,
                    user=user,
                )
            except NoResultFound:
                # Fallback to secondary storage
                try:
                    existing_doc = SecondaryDocument.read(
                        db_session=session,
                        identifier=doc_identifier,
                        user=user,
                    )
                except NoResultFound:
                    raise ValueError(
                        f"Document with id {doc_identifier} does not exist."
                    )

            # Update the document's properties
            update_fields = updated_document.model_dump(
                exclude_unset=True, exclude_none=True
            )
            for field, value in update_fields.items():
                setattr(existing_doc, field, value)

            # Commit the changes
            existing_doc.update(session, user=user)
            return existing_doc.to_pydantic()

    @enforce_types
    def remove_document_by_identifier(
        self, doc_identifier: str, user: PydanticUser
    ) -> bool:
        """Delete a document from either primary or secondary storage."""
        if not doc_identifier:
            raise ValueError("Document identifier is required.")

        with self.session_creator() as session:
            # Attempt to fetch from primary storage
            try:
                document = PrimaryDocument.read(
                    db_session=session, identifier=doc_identifier, user=user
                )
                document.hard_delete(session, user=user)
                return True
            except NoResultFound:
                # Fallback to secondary storage
                try:
                    document = SecondaryDocument.read(
                        db_session=session, identifier=doc_identifier, user=user
                    )
                    document.hard_delete(session, user=user)
                    return True
                except NoResultFound:
                    raise NoResultFound(f"Document with id {doc_identifier} not found.")

    def remove_documents(
        self,
        user: PydanticUser,
        documents: List[PydanticDocument],
    ) -> bool:
        # TODO: This is inefficient
        # TODO: Consider implementing a bulk delete function
        for doc in documents:
            self.remove_document_by_identifier(doc_identifier=doc.id, user=user)
        return True
