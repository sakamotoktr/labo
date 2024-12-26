import os
from typing import List, Optional


class BlockService:
    """Service class to manage operations related to Blocks."""

    def __init__(self):
        # Initializing the database context similarly as in ToolService
        self.db_session_factory = db_context

    @enforce_types
    def save_or_modify_block(self, block: SchemaBlock, user: SchemaUser) -> SchemaBlock:
        """Save or modify a block based on the SchemaBlock schema."""
        existing_block = self.fetch_block_by_id(block.id, user)
        if existing_block:
            update_info = SchemaBlockUpdate(**block.model_dump(exclude_none=True))
            self.modify_block(block.id, update_info, user)
        else:
            with self.db_session_factory() as session:
                block_data = block.model_dump(exclude_none=True)
                new_block = ORMBlock(**block_data, organization_id=user.organization_id)
                new_block.create(session, actor=user)
            return new_block.to_pydantic()

    @enforce_types
    def modify_block(
        self, block_id: str, block_update: SchemaBlockUpdate, user: SchemaUser
    ) -> SchemaBlock:
        """Modify a block by its ID with the given SchemaBlockUpdate object."""
        # Ensure block exists

        with self.db_session_factory() as session:
            block = ORMBlock.read(db_session=session, identifier=block_id, actor=user)
            update_info = block_update.model_dump(exclude_unset=True, exclude_none=True)

            for key, value in update_info.items():
                setattr(block, key, value)

            block.update(db_session=session, actor=user)
            return block.to_pydantic()

    @enforce_types
    def remove_block(self, block_id: str, user: SchemaUser) -> SchemaBlock:
        """Remove a block by its ID."""
        with self.db_session_factory() as session:
            block = ORMBlock.read(db_session=session, identifier=block_id)
            block.hard_delete(db_session=session, actor=user)
            return block.to_pydantic()

    @enforce_types
    def retrieve_blocks(
        self,
        user: SchemaUser,
        label: Optional[str] = None,
        is_template: Optional[bool] = None,
        template_name: Optional[str] = None,
        id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[SchemaBlock]:
        """Retrieve blocks based on various optional filters."""
        with self.db_session_factory() as session:
            # Set up filters
            filter_criteria = {"organization_id": user.organization_id}
            if label:
                filter_criteria["label"] = label
            if is_template is not None:
                filter_criteria["is_template"] = is_template
            if template_name:
                filter_criteria["template_name"] = template_name
            if id:
                filter_criteria["id"] = id

            blocks = ORMBlock.list(
                db_session=session, cursor=cursor, limit=limit, **filter_criteria
            )

            return [block.to_pydantic() for block in blocks]

    @enforce_types
    def fetch_block_by_id(
        self, block_id: str, user: Optional[SchemaUser] = None
    ) -> Optional[SchemaBlock]:
        """Fetch a block by its ID."""
        with self.db_session_factory() as session:
            try:
                block = ORMBlock.read(
                    db_session=session, identifier=block_id, actor=user
                )
                return block.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def fetch_all_blocks_by_ids(
        self, block_ids: List[str], user: Optional[SchemaUser] = None
    ) -> List[SchemaBlock]:
        # TODO: Optimize by listing instead of individual queries per block_id
        blocks = []
        for block_id in block_ids:
            block = self.fetch_block_by_id(block_id, user=user)
            blocks.append(block)
        return blocks

    @enforce_types
    def insert_default_blocks(self, user: SchemaUser):
        for persona_file in list_persona_files():
            content = open(persona_file, "r", encoding="utf-8").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            self.save_or_modify_block(
                SchemaPersona(template_name=name, value=content, is_template=True),
                user=user,
            )

        for human_file in list_human_files():
            content = open(human_file, "r", encoding="utf-8").read()
            name = os.path.basename(human_file).replace(".txt", "")
            self.save_or_modify_block(
                SchemaHuman(template_name=name, value=content, is_template=True),
                user=user,
            )
