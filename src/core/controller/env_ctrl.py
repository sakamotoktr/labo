from pathlib import Path
from typing import Dict, List, Optional

log = initialize_logger(__name__)


class ConfigHandler:
    """Handles all business logic related to Sandbox configurations and environment variables."""

    def __init__(self, config_data):
        self.session_factory = database_context

    @enforce_type_checks
    def fetch_or_initialize_default_sandbox_config(
        self, sandbox_category: SandboxCategory, user: PydanticUser
    ) -> PydanticSandboxConfig:
        sandbox_config = self.fetch_sandbox_config_by_category(
            sandbox_category, user=user
        )
        if not sandbox_config:
            log.debug(
                f"Initializing sandbox config for {sandbox_category}, none found for organization {user.organization_id}."
            )

            # TODO: Extend sandbox categories in the future
            if sandbox_category == SandboxCategory.E2B:
                default_config = {}  # Empty
            else:
                # TODO: Consider moving this to environment variables instead of saving in the database
                local_sandbox_path = str(
                    Path(__file__).parent / "tool_sandbox_environment"
                )
                default_config = LocalSandboxConfig(
                    sandbox_dir=local_sandbox_path
                ).model_dump(exclude_none=True)

            sandbox_config = self.create_or_modify_sandbox_config(
                SandboxConfigCreate(config=default_config), user=user
            )
        return sandbox_config

    @enforce_type_checks
    def create_or_modify_sandbox_config(
        self, sandbox_config_creation: SandboxConfigCreate, user: PydanticUser
    ) -> PydanticSandboxConfig:
        """Create or modify a sandbox config using the PydanticSandboxConfig schema."""
        config_data = sandbox_config_creation.config
        sandbox_category = config_data.type
        new_sandbox_config = PydanticSandboxConfig(
            type=sandbox_category,
            config=config_data.model_dump(exclude_none=True),
            organization_id=user.organization_id,
        )

        # Try to retrieve an existing sandbox config by category within the organization
        existing_sandbox = self.fetch_sandbox_config_by_category(
            new_sandbox_config.type, user=user
        )
        if existing_sandbox:
            # Prepare update data, excluding fields that should not be reset
            update_data = new_sandbox_config.model_dump(
                exclude_unset=True, exclude_none=True
            )
            update_data = {
                key: value
                for key, value in update_data.items()
                if getattr(existing_sandbox, key) != value
            }

            # If changes are detected, update the existing sandbox config
            if update_data:
                existing_sandbox = self.modify_sandbox_config(
                    existing_sandbox.id, SandboxConfigUpdate(**update_data), user
                )
            else:
                print_debug(
                    f"`create_or_modify_sandbox_config` was called with user_id={user.id}, organization_id={user.organization_id}, "
                    f"type={new_sandbox_config.type}, but no changes found to update."
                )

            return existing_sandbox
        else:
            # Create a new sandbox config if none exists
            with self.session_factory() as session:
                new_sandbox_db = SandboxConfigModel(
                    **new_sandbox_config.model_dump(exclude_none=True)
                )
                new_sandbox_db.create(session, user=user)
                return new_sandbox_db.to_pydantic()

    @enforce_type_checks
    def modify_sandbox_config(
        self,
        sandbox_config_identifier: str,
        sandbox_update_data: SandboxConfigUpdate,
        user: PydanticUser,
    ) -> PydanticSandboxConfig:
        """Update an existing sandbox config."""
        with self.session_factory() as session:
            sandbox = SandboxConfigModel.read(
                db_session=session, identifier=sandbox_config_identifier, user=user
            )
            if sandbox.type != sandbox_update_data.config.type:
                raise ValueError(
                    f"Sandbox type mismatch: attempted to update sandbox of type {sandbox.type} with config of type {sandbox_update_data.config.type}"
                )

            update_fields = sandbox_update_data.model_dump(
                exclude_unset=True, exclude_none=True
            )
            update_fields = {
                key: value
                for key, value in update_fields.items()
                if getattr(sandbox, key) != value
            }

            if update_fields:
                for key, value in update_fields.items():
                    setattr(sandbox, key, value)
                sandbox.update(db_session=session, user=user)
            else:
                print_debug(
                    f"`modify_sandbox_config` called with user_id={user.id}, organization_id={user.organization_id}, "
                    f"name={sandbox.type}, but no changes found to update."
                )
            return sandbox.to_pydantic()

    @enforce_type_checks
    def remove_sandbox_config(
        self, sandbox_config_identifier: str, user: PydanticUser
    ) -> PydanticSandboxConfig:
        """Remove a sandbox config by its ID."""
        with self.session_factory() as session:
            sandbox = SandboxConfigModel.read(
                db_session=session, identifier=sandbox_config_identifier, user=user
            )
            sandbox.hard_delete(db_session=session, user=user)
            return sandbox.to_pydantic()

    @enforce_type_checks
    def list_sandbox_configs(
        self,
        user: PydanticUser,
        pagination_token: Optional[str] = None,
        page_limit: Optional[int] = 50,
    ) -> List[PydanticSandboxConfig]:
        """Retrieve a list of sandbox configs with optional pagination."""
        with self.session_factory() as session:
            sandboxes = SandboxConfigModel.list(
                db_session=session,
                cursor=pagination_token,
                limit=page_limit,
                organization_id=user.organization_id,
            )
            return [sandbox.to_pydantic() for sandbox in sandboxes]

    @enforce_type_checks
    def fetch_sandbox_config_by_id(
        self, sandbox_config_identifier: str, user: Optional[PydanticUser] = None
    ) -> Optional[PydanticSandboxConfig]:
        """Retrieve a sandbox config by its ID."""
        with self.session_factory() as session:
            try:
                sandbox = SandboxConfigModel.read(
                    db_session=session, identifier=sandbox_config_identifier, user=user
                )
                return sandbox.to_pydantic()
            except NoResultFound:
                return None

    @enforce_type_checks
    def fetch_sandbox_config_by_category(
        self, category: SandboxCategory, user: Optional[PydanticUser] = None
    ) -> Optional[PydanticSandboxConfig]:
        """Retrieve a sandbox config by its category."""
        with self.session_factory() as session:
            try:
                sandboxes = SandboxConfigModel.list(
                    db_session=session,
                    category=category,
                    organization_id=user.organization_id,
                    limit=1,
                )
                if sandboxes:
                    return sandboxes[0].to_pydantic()
                return None
            except NoResultFound:
                return None

    @enforce_type_checks
    def create_sandbox_env_variable(
        self,
        env_var_creation: SandboxEnvironmentVariableCreate,
        sandbox_config_identifier: str,
        user: PydanticUser,
    ) -> PydanticEnvVar:
        """Create a new environment variable for the sandbox."""
        env_var = PydanticEnvVar(
            **env_var_creation.model_dump(),
            sandbox_config_id=sandbox_config_identifier,
            organization_id=user.organization_id,
        )

        existing_env_var = self.fetch_sandbox_env_var_by_key_and_sandbox_id(
            env_var.key, env_var.sandbox_config_id, user=user
        )
        if existing_env_var:
            update_fields = env_var.model_dump(exclude_unset=True, exclude_none=True)
            update_fields = {
                key: value
                for key, value in update_fields.items()
                if getattr(existing_env_var, key) != value
            }
            if update_fields:
                existing_env_var = self.modify_sandbox_env_var(
                    existing_env_var.id,
                    SandboxEnvironmentVariableUpdate(**update_fields),
                    user,
                )
            else:
                print_debug(
                    f"`create_or_update_sandbox_env_var` was called with user_id={user.id}, organization_id={user.organization_id}, "
                    f"key={env_var.key}, but no updates were needed."
                )

            return existing_env_var
        else:
            with self.session_factory() as session:
                new_env_var = SandboxEnvVarModel(
                    **env_var.model_dump(exclude_none=True)
                )
                new_env_var.create(session, user=user)
            return new_env_var.to_pydantic()

    @enforce_type_checks
    def modify_sandbox_env_var(
        self,
        env_var_identifier: str,
        env_var_update_data: SandboxEnvironmentVariableUpdate,
        user: PydanticUser,
    ) -> PydanticEnvVar:
        """Update an existing environment variable in the sandbox."""
        with self.session_factory() as session:
            env_var = SandboxEnvVarModel.read(
                db_session=session, identifier=env_var_identifier, user=user
            )
            update_fields = env_var_update_data.model_dump(
                exclude_unset=True, exclude_none=True
            )
            update_fields = {
                key: value
                for key, value in update_fields.items()
                if getattr(env_var, key) != value
            }

            if update_fields:
                for key, value in update_fields.items():
                    setattr(env_var, key, value)
                env_var.update(db_session=session, user=user)
            else:
                print_debug(
                    f"`modify_sandbox_env_var` called with user_id={user.id}, organization_id={user.organization_id}, "
                    f"key={env_var.key}, but no updates detected."
                )
            return env_var.to_pydantic()

    @enforce_type_checks
    def remove_sandbox_env_var(
        self, env_var_identifier: str, user: PydanticUser
    ) -> PydanticEnvVar:
        """Remove an environment variable by its ID."""
        with self.session_factory() as session:
            env_var = SandboxEnvVarModel.read(
                db_session=session, identifier=env_var_identifier, user=user
            )
            env_var.hard_delete(db_session=session, user=user)
            return env_var.to_pydantic()

    @enforce_type_checks
    def list_sandbox_env_vars(
        self,
        sandbox_config_identifier: str,
        user: PydanticUser,
        cursor_token: Optional[str] = None,
        page_limit: Optional[int] = 50,
    ) -> List[PydanticEnvVar]:
        """Retrieve a list of sandbox environment variables with optional pagination."""
        with self.session_factory() as session:
            env_vars = SandboxEnvVarModel.list(
                db_session=session,
                cursor=cursor_token,
                limit=page_limit,
                organization_id=user.organization_id,
                sandbox_config_id=sandbox_config_identifier,
            )
            return [env_var.to_pydantic() for env_var in env_vars]

    @enforce_type_checks
    def list_sandbox_env_vars_by_key(
        self,
        search_key: str,
        user: PydanticUser,
        cursor_token: Optional[str] = None,
        page_limit: Optional[int] = 50,
    ) -> List[PydanticEnvVar]:
        """Retrieve a list of sandbox environment variables by their key."""
        with self.session_factory() as session:
            env_vars = SandboxEnvVarModel.list(
                db_session=session,
                cursor=cursor_token,
                limit=page_limit,
                organization_id=user.organization_id,
                key=search_key,
            )
            return [env_var.to_pydantic() for env_var in env_vars]

    @enforce_type_checks
    def get_sandbox_env_vars_as_map(
        self,
        sandbox_config_identifier: str,
        user: PydanticUser,
        cursor_token: Optional[str] = None,
        page_limit: Optional[int] = 50,
    ) -> Dict[str, str]:
        env_vars = self.list_sandbox_env_vars(
            sandbox_config_identifier, user, cursor_token, page_limit
        )
        result_map = {}
        for env_var in env_vars:
            result_map[env_var.key] = env_var.value
        return result_map

    @enforce_type_checks
    def fetch_sandbox_env_var_by_key_and_sandbox_id(
        self,
        key: str,
        sandbox_config_identifier: str,
        user: Optional[PydanticUser] = None,
    ) -> Optional[PydanticEnvVar]:
        """Retrieve an environment variable by its key and sandbox_config_identifier."""
        with self.session_factory() as session:
            try:
                env_vars = SandboxEnvVarModel.list(
                    db_session=session,
                    key=key,
                    sandbox_config_id=sandbox_config_identifier,
                    organization_id=user.organization_id,
                    limit=1,
                )
                if env_vars:
                    return env_vars[0].to_pydantic()
                return None
            except NoResultFound:
                return None
