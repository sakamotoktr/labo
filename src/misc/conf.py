import configparser
import os
from dataclasses import dataclass
from typing import Optional

system_logger = get_logger(__name__)


def extract_setting(cfg_obj, category, setting_name):
    return (
        cfg_obj.get(category, setting_name)
        if category in cfg_obj and cfg_obj.has_option(category, setting_name)
        else None
    )


def update_setting(cfg_obj, category, setting_name, setting_value):
    if setting_value is not None:
        if category not in cfg_obj:
            cfg_obj.add_section(category)
        cfg_obj.set(category, setting_name, setting_value)


@dataclass
class SystemConfiguration:
    settings_location: str = os.getenv("MEMGPT_CONFIG_PATH") or os.path.join(
        LABO_DIR, "config"
    )
    template: str = DEFAULT_PRESET
    character: str = DEFAULT_PERSONA
    user_profile: str = DEFAULT_HUMAN

    archive_db_type: str = "sqlite"
    archive_db_location: str = LABO_DIR
    archive_db_connection: str = None

    memory_db_type: str = "sqlite"
    memory_db_location: str = LABO_DIR
    memory_db_connection: str = None

    info_storage_type: str = "sqlite"
    info_storage_location: str = LABO_DIR
    info_storage_connection: str = None

    state_manager_type: str = None
    state_file_path: str = None
    state_db_uri: str = None

    system_version: str = labo.__version__
    terms_accepted: bool = False
    persona_memory_limit: int = CORE_MEMORY_PERSONA_CHAR_LIMIT
    user_memory_limit: int = CORE_MEMORY_HUMAN_CHAR_LIMIT

    def __post_init__(self):
        pass

    @classmethod
    def initialize(
        cls,
        llm_settings: Optional[LLMConfig] = None,
        embed_settings: Optional[EmbeddingConfig] = None,
    ) -> "SystemConfiguration":
        cfg_parser = configparser.ConfigParser()
        target_path = os.getenv("MEMGPT_CONFIG_PATH") or cls.settings_location

        cls.setup_directories()
        printd(f"Reading configuration from: {target_path}")

        if os.path.exists(target_path):
            cfg_parser.read(target_path)

            settings = {
                "template": extract_setting(cfg_parser, "defaults", "preset"),
                "character": extract_setting(cfg_parser, "defaults", "persona"),
                "user_profile": extract_setting(cfg_parser, "defaults", "human"),
                "archive_db_type": extract_setting(
                    cfg_parser, "archival_storage", "type"
                ),
                "archive_db_location": extract_setting(
                    cfg_parser, "archival_storage", "path"
                ),
                "archive_db_connection": extract_setting(
                    cfg_parser, "archival_storage", "uri"
                ),
                "memory_db_type": extract_setting(cfg_parser, "recall_storage", "type"),
                "memory_db_location": extract_setting(
                    cfg_parser, "recall_storage", "path"
                ),
                "memory_db_connection": extract_setting(
                    cfg_parser, "recall_storage", "uri"
                ),
                "info_storage_type": extract_setting(
                    cfg_parser, "metadata_storage", "type"
                ),
                "info_storage_location": extract_setting(
                    cfg_parser, "metadata_storage", "path"
                ),
                "info_storage_connection": extract_setting(
                    cfg_parser, "metadata_storage", "uri"
                ),
                "settings_location": target_path,
                "system_version": extract_setting(
                    cfg_parser, "version", "labo_version"
                ),
            }
            settings = {k: v for k, v in settings.items() if v is not None}
            return cls(**settings)

        new_config = cls(settings_location=target_path)
        new_config.setup_directories()
        return new_config

    def persist(self):
        cfg_parser = configparser.ConfigParser()

        update_setting(cfg_parser, "defaults", "preset", self.template)
        update_setting(cfg_parser, "defaults", "persona", self.character)
        update_setting(cfg_parser, "defaults", "human", self.user_profile)

        update_setting(cfg_parser, "archival_storage", "type", self.archive_db_type)
        update_setting(cfg_parser, "archival_storage", "path", self.archive_db_location)
        update_setting(
            cfg_parser, "archival_storage", "uri", self.archive_db_connection
        )

        update_setting(cfg_parser, "recall_storage", "type", self.memory_db_type)
        update_setting(cfg_parser, "recall_storage", "path", self.memory_db_location)
        update_setting(cfg_parser, "recall_storage", "uri", self.memory_db_connection)

        update_setting(cfg_parser, "metadata_storage", "type", self.info_storage_type)
        update_setting(
            cfg_parser, "metadata_storage", "path", self.info_storage_location
        )
        update_setting(
            cfg_parser, "metadata_storage", "uri", self.info_storage_connection
        )

        update_setting(cfg_parser, "version", "labo_version", labo.__version__)

        self.setup_directories()

        with open(self.settings_location, "w", encoding="utf-8") as config_file:
            cfg_parser.write(config_file)
        system_logger.debug(f"Configuration saved to: {self.settings_location}")

    @staticmethod
    def is_configured():
        target_path = (
            os.getenv("MEMGPT_CONFIG_PATH") or SystemConfiguration.settings_location
        )
        assert not os.path.isdir(
            target_path
        ), f"Invalid configuration path: {target_path} (directory not allowed)"
        return os.path.exists(target_path)

    @staticmethod
    def setup_directories():
        if not os.path.exists(LABO_DIR):
            os.makedirs(LABO_DIR, exist_ok=True)

        required_folders = [
            "personas",
            "humans",
            "archival",
            "agents",
            "functions",
            "system_prompts",
            "presets",
            "settings",
        ]

        for folder_name in required_folders:
            folder_path = os.path.join(LABO_DIR, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
