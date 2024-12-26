import uuid
from datetime import datetime
from logging import getLogger
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Create a logger instance for the current module
logger = getLogger(__name__)


class LABOBase(BaseModel):
    """
    The `LABOBase` class serves as a base model for other models in the application.
    It provides a set of configurations and methods related to generating and validating unique identifiers.

    Class Configuration:
    - `model_config`: A `ConfigDict` object that configures how the model behaves during initialization,
                      deserialization, and serialization.
        - `populate_by_name`: When set to `True`, allows populating model fields using attribute names instead of
                              just field aliases. This provides flexibility in how data is assigned to the model.
        - `from_attributes`: When set to `True`, enables the model to be populated from object attributes during
                             initialization. This is useful when working with ORM objects or other data sources
                             where attributes need to be mapped to model fields.
        - `extra`: Set to `"forbid"`, this ensures that any additional fields not defined in the model are not
                   allowed during deserialization. This helps in maintaining data integrity and avoiding
                   unexpected behavior due to extra or unknown fields.

    Methods:
    - `generate_id_field`: A class method that generates a `Field` object for the unique identifier of the model.
                           It takes an optional `prefix` parameter, which defaults to the class's `__id_prefix__`
                           if not provided. The generated `Field` object has specific properties like a description,
                           a regex pattern for validation, examples, and a default factory method to generate the ID.
    - `_generate_id`: A private class method that actually generates the unique ID string. It takes a `prefix`
                      (which again defaults to `__id_prefix__` if not given) and combines it with a UUID to create
                      a formatted ID in the format "{prefix}-{uuid}".
    - `_id_regex_pattern`: A private class method that constructs a regular expression pattern for validating
                          the unique ID. The pattern ensures that the ID starts with the given `prefix` followed by
                          a specific UUID format.
    - `_id_example`: A private class method that returns an example ID string in the correct format for the given
                     `prefix`. This can be useful for documentation or for showing expected ID formats.
    - `_id_description`: A private class method that generates a human-readable description of the unique ID,
                         indicating which type of object it belongs to based on the `prefix`.
    - `allow_bare_uuids`: A field validator method for the `id` field. It checks if the provided value is a `UUID`
                          instance. If so, it logs a deprecation warning suggesting the use of the full prefixed ID
                          and then converts the bare UUID into the prefixed format. Otherwise, it returns the value
                          unchanged. This helps in migrating from a previous system that might have used bare UUIDs
                          to the current ID format with prefixes.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        extra="forbid",
    )

    @classmethod
    def generate_id_field(cls, prefix: Optional[str] = None) -> "Field":
        """
        Generate a `Field` object for the unique identifier of the model.

        This method creates a `Field` with specific properties that are used to define and validate the unique
        identifier for instances of the model. The `Field` includes a description, a regex pattern for validation,
        examples, and a default factory method to generate the ID.

        Args:
        - `prefix`: An optional string representing the prefix to be used for the ID. If not provided, it defaults
                    to the class's `__id_prefix__`.

        Returns:
        - `Field`: A `Field` object configured for the unique identifier of the model.
        """
        prefix = prefix or cls.__id_prefix__

        return Field(
           ...,
            description=cls._id_description(prefix),
            pattern=cls._id_regex_pattern(prefix),
            examples=[cls._id_example(prefix)],
            default_factory=cls._generate_id,
        )

    @classmethod
    def _generate_id(cls, prefix: Optional[str] = None) -> str:
        """
        Generate a unique ID string for the model.

        This private class method combines the given `prefix` (or the default `__id_prefix__` if not provided)
        with a generated UUID to create a formatted ID string.

        Args:
        - `prefix`: An optional string representing the prefix for the ID. Defaults to `__id_prefix__` if not given.

        Returns:
        - `str`: A unique ID string in the format "{prefix}-{uuid}".
        """
        prefix = prefix or cls.__id_prefix__
        return f"{prefix}-{uuid.uuid4()}"

    @classmethod
    def _id_regex_pattern(cls, prefix: str):
        """
        Generate a regular expression pattern for validating the unique ID of the model.

        The constructed pattern ensures that the ID starts with the specified `prefix` followed by a valid UUID
        format.

        Args:
        - `prefix`: A string representing the prefix for the ID.

        Returns:
        - `str`: A regular expression pattern for validating the ID.
        """
        return (
            r"^" + prefix + r"-"
            r"[a-fA-F0-9]{8}"
            r"-[a-fA-F0-9]{4}"
            r"-[a-fA-F0-9]{4}"
            r"-[a-fA-F0-9]{4}"
            r"-[a-fA-F0-9]{12}"
        )

    @classmethod
    def _id_example(cls, prefix: str):
        """
        Generate an example ID string for the model.

        This method returns an example ID in the correct format for the given `prefix`, which can be useful for
        documentation or demonstrating the expected ID format.

        Args:
        - `prefix`: A string representing the prefix for the ID.

        Returns:
        - `str`: An example ID string in the format "{prefix}-{uuid}".
        """
        return f"{prefix}-123e4567-e89b-12d3-a456-426614174000"

    @classmethod
    def _id_description(cls, prefix: str):
        """
        Generate a human-readable description for the unique ID of the model.

        The description indicates which type of object the ID belongs to, based on the `prefix` used.

        Args:
        - `prefix`: A string representing the prefix for the ID.

        Returns:
        - `str`: A human-readable description of the ID.
        """
        return f"The human-friendly ID of the {prefix.capitalize()}"

    @field_validator("id", check_fields=False, mode="before")
    @classmethod
    def allow_bare_uuids(cls, v, values):
        """
        Field validator for the `id` field to handle bare UUIDs.

        This method checks if the provided value for the `id` field is a `UUID` instance. If it is, it logs a
        deprecation warning suggesting the use of the full prefixed ID and then converts the bare UUID into the
        prefixed format. If the value is not a `UUID`, it simply returns the value unchanged.

        Args:
        - `v`: The value provided for the `id` field.
        - `values`: A dictionary containing the values of other fields in the model (not used in this method but
                    required by the `field_validator` decorator).

        Returns:
        - `str`: The processed value for the `id` field, either converted to the prefixed format if it was a `UUID`
                 or the original value if not.
        """
        _ = values
        if isinstance(v, UUID):
            logger.debug(f"Bare UUIDs are deprecated, please use the full prefixed id ({cls.__id_prefix__})!")
            return f"{cls.__id_prefix__}-{v}"
        return v


class OrmMetadataBase(LABOBase):
    """
    The `OrmMetadataBase` class extends the `LABOBase` class to add metadata related to object creation and updates.
    It's likely designed to be used as a base for models that interact with an ORM system, providing additional
    information about who created and last updated an object, as well as the timestamps for those events.

    Attributes:
    - `created_by_id`: An optional string representing the ID of the user who created the object. This can be used
                       for auditing purposes or to track the originator of an object in the system.
    - `last_updated_by_id`: An optional string representing the ID of the user who last updated the object. Similar
                           to `created_by_id`, it helps in tracking changes made to the object over time.
    - `created_at`: An optional `datetime` object representing the timestamp when the object was created. This
                    provides a record of the exact time when the object first entered the system.
    - `updated_at`: An optional `datetime` object representing the timestamp when the object was last updated.
                    It's useful for knowing when the most recent changes were made to the object.
    """
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this object.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that made this object.")
    created_at: Optional[datetime] = Field(None, description="The timestamp when the object was created.")
    updated_at: Optional[datetime] = Field(None, description="The timestamp when the object was last updated.")