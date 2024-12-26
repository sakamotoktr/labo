from typing import List, Optional, Tuple


class AccountManager:
    """Class responsible for handling user-related operations."""

    DEFAULT_ACCOUNT_NAME = "standard_user"
    DEFAULT_ACCOUNT_ID = "user-00000000-0000-4000-8000-000000000000"

    def __init__(self):
        # Initializing database context from server
        self.db_session_factory = db_context

    @enforce_types
    def initialize_default_account(
        self, organization_id: str = OrganizationHandler.DEFAULT_ORG_ID
    ) -> PydanticAccount:
        """Initialize the default account for a given organization."""
        with self.db_session_factory() as session:
            # Verifying if the provided organization ID exists
            try:
                OrganizationEntity.fetch_by_id(
                    db_session=session, identifier=organization_id
                )
            except NoResultFound:
                raise ValueError(
                    f"Organization with ID {organization_id} does not exist."
                )

            # Attempt to find the default account
            try:
                account = AccountEntity.fetch_by_id(
                    db_session=session, identifier=self.DEFAULT_ACCOUNT_ID
                )
            except NoResultFound:
                # If not found, create the default account
                account = AccountEntity(
                    id=self.DEFAULT_ACCOUNT_ID,
                    name=self.DEFAULT_ACCOUNT_NAME,
                    organization_id=organization_id,
                )
                account.save(session)

            return account.to_pydantic()

    @enforce_types
    def register_account(self, account_data: PydanticAccount) -> PydanticAccount:
        """Register a new account if not already present."""
        with self.db_session_factory() as session:
            new_account = AccountEntity(**account_data.model_dump())
            new_account.save(session)
            return new_account.to_pydantic()

    @enforce_types
    def modify_account(self, account_update: AccountModification) -> PydanticAccount:
        """Modify the details of an existing account."""
        with self.db_session_factory() as session:
            # Fetch the current account by its ID
            current_account = AccountEntity.fetch_by_id(
                db_session=session, identifier=account_update.id
            )

            # Update the account fields based on the provided modification data
            updated_fields = account_update.model_dump(
                exclude_unset=True, exclude_none=True
            )
            for field, value in updated_fields.items():
                setattr(current_account, field, value)

            # Save the modified account back to the database
            current_account.save(session)
            return current_account.to_pydantic()

    @enforce_types
    def remove_account_by_id(self, account_id: str):
        """Remove an account and its related data (agents, sources, mappings)."""
        with self.db_session_factory() as session:
            # Delete the account from the database
            account = AccountEntity.fetch_by_id(
                db_session=session, identifier=account_id
            )
            account.delete_permanently(session)

            session.commit()

    @enforce_types
    def fetch_account_by_id(self, account_id: str) -> PydanticAccount:
        """Retrieve an account by its unique ID."""
        with self.db_session_factory() as session:
            account = AccountEntity.fetch_by_id(
                db_session=session, identifier=account_id
            )
            return account.to_pydantic()

    @enforce_types
    def fetch_default_account(self) -> PydanticAccount:
        """Retrieve the default account."""
        return self.fetch_account_by_id(self.DEFAULT_ACCOUNT_ID)

    @enforce_types
    def fetch_account_or_default(self, account_id: Optional[str] = None):
        """Retrieve a specific account or fall back to the default account."""
        if not account_id:
            return self.fetch_default_account()

        try:
            return self.fetch_account_by_id(account_id=account_id)
        except NoResultFound:
            return self.fetch_default_account()

    @enforce_types
    def get_all_accounts(
        self, pagination_cursor: Optional[str] = None, page_size: Optional[int] = 50
    ) -> Tuple[Optional[str], List[PydanticAccount]]:
        """Fetch all accounts with pagination (cursor-based)."""
        with self.db_session_factory() as session:
            accounts = AccountEntity.list_all(
                db_session=session, cursor=pagination_cursor, limit=page_size
            )
            return [account.to_pydantic() for account in accounts]
