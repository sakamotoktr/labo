import configparser
import os
from dataclasses import dataclass
from typing import Optional

VALID_AUTH_METHODS = ["token_auth", "key_auth"]


def get_config_value(cfg, section, key):
    try:
        return cfg[section][key]
    except:
        return None


def update_config_section(cfg, section, key, val):
    if val is not None:
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg[section][key] = str(val)


@dataclass
class ServiceAuthConfig:
    config_location: str = os.path.join(LABO_DIR, "auth_settings")

    # cloud provider settings
    cloud_a_auth_method: str = "token_auth"
    cloud_a_token: Optional[str] = os.getenv("OPENAI_API_KEY")

    cloud_b_token: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    cloud_c_token: Optional[str] = None
    cloud_c_endpoint: Optional[str] = None

    cloud_d_auth_method: str = "key_auth"
    cloud_d_token: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")

    cloud_e_token: Optional[str] = None

    cloud_f_token: Optional[str] = os.getenv("GROQ_API_KEY")

    cloud_g_token: Optional[str] = None

    # service configuration
    service_ver: Optional[str] = None
    service_url: Optional[str] = None
    service_name: Optional[str] = None

    embedding_ver: Optional[str] = None
    embedding_url: Optional[str] = None
    embedding_name: Optional[str] = None

    # custom configuration
    custom_auth_method: Optional[str] = None
    custom_token: Optional[str] = None

    def store(self):
        cfg = configparser.ConfigParser()

        # provider configurations
        for provider, settings in {
            "cloud_a": {
                "auth_method": self.cloud_a_auth_method,
                "token": self.cloud_a_token,
            },
            "cloud_b": {"token": self.cloud_b_token},
            "cloud_c": {
                "token": self.cloud_c_token,
                "endpoint": self.cloud_c_endpoint,
            },
            "cloud_d": {
                "auth_method": self.cloud_d_auth_method,
                "token": self.cloud_d_token,
                "ver": self.service_ver,
                "url": self.service_url,
                "name": self.service_name,
                "embed_ver": self.embedding_ver,
                "embed_url": self.embedding_url,
                "embed_name": self.embedding_name,
            },
            "cloud_e": {"token": self.cloud_e_token},
            "cloud_f": {"token": self.cloud_f_token},
            "cloud_g": {"token": self.cloud_g_token},
            "custom": {
                "auth_method": self.custom_auth_method,
                "token": self.custom_token,
            },
        }.items():
            for key, value in settings.items():
                update_config_section(cfg, provider, key, value)

        os.makedirs(LABO_DIR, exist_ok=True)
        with open(self.config_location, "w", encoding="utf-8") as f:
            cfg.write(f)

    @classmethod
    def retrieve(cls) -> "ServiceAuthConfig":
        cfg = configparser.ConfigParser()
        config_path = os.getenv("MEMGPT_CREDENTIALS_PATH", cls.config_location)

        if not os.path.exists(config_path):
            return cls(config_location=config_path).store()

        cfg.read(config_path)
        settings = {
            "cloud_a_auth_method": get_config_value(cfg, "cloud_a", "auth_method"),
            "cloud_a_token": get_config_value(cfg, "cloud_a", "token"),
            "cloud_b_token": get_config_value(cfg, "cloud_b", "token"),
            "cloud_c_token": get_config_value(cfg, "cloud_c", "token"),
            "cloud_d_auth_method": get_config_value(cfg, "cloud_d", "auth_method"),
            "cloud_d_token": get_config_value(cfg, "cloud_d", "token"),
            "service_ver": get_config_value(cfg, "cloud_d", "ver"),
            "service_url": get_config_value(cfg, "cloud_d", "url"),
            "service_name": get_config_value(cfg, "cloud_d", "name"),
            "embedding_ver": get_config_value(cfg, "cloud_d", "embed_ver"),
            "embedding_url": get_config_value(cfg, "cloud_d", "embed_url"),
            "embedding_name": get_config_value(cfg, "cloud_d", "embed_name"),
            "cloud_e_token": get_config_value(cfg, "cloud_e", "token"),
            "cloud_f_token": get_config_value(cfg, "cloud_f", "token"),
            "cloud_g_token": get_config_value(cfg, "cloud_g", "token"),
            "custom_auth_method": get_config_value(cfg, "custom", "auth_method"),
            "custom_token": get_config_value(cfg, "custom", "token"),
            "config_location": config_path,
        }
        return cls(**{k: v for k, v in settings.items() if v is not None})

    @staticmethod
    def is_configured():
        config_path = os.getenv(
            "MEMGPT_CREDENTIALS_PATH", ServiceAuthConfig.config_location
        )
        if os.path.isdir(config_path):
            raise ValueError(
                f"Configuration path {config_path} must not be a directory"
            )
        return os.path.exists(config_path)
