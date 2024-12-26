from typing import Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from labo.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from labo.schemas.labo_base import LABOBase


class BaseBlock(LABOBase, validate_assignment=True):
    """
    Represents the base structure for a block in the system, which serves as a foundational model for more specific
    block types. It includes attributes related to the block's content, character limit, template status, and metadata.

    Attributes:
    - `__id_prefix__`: A prefix used for generating the block's ID, set to "block". This likely helps in identifying
                       and categorizing the block within a larger system where IDs follow a specific naming convention.
    - `value`: The main text content of the block. This is a required field and holds the actual data that the block
               represents, such as a piece of text for a specific role or context.
    - `limit`: An integer representing the character limit of the block. By default, it uses the value from
               `CORE_MEMORY_BLOCK_CHAR_LIMIT` which is likely a predefined constant in the `labo.constants` module.
               This limit is used to control the maximum length of the `value` attribute.
    - `template_name`: An optional string representing the name of the block if it is a template. This can be used to
                       identify and reuse specific block templates, and it has an alias "name" which might be used
                       for serialization/deserialization purposes.
    - `is_template`: A boolean indicating whether the block is a template. Templates could be predefined options for
                     things like saved human/persona configurations that can be reused in different contexts.
    - `label`: An optional string representing a label for the block. This could be used to categorize or identify
               the block in a specific context, such as indicating whether it's related to a "human" or "persona" role.
    - `description`: An optional string providing a more detailed description of the block. This can offer additional
                     information about its purpose or usage.
    - `metadata_`: An optional dictionary containing additional metadata for the block. This could include custom
                   tags, user-defined settings, or other information relevant to the block's operation.

    Class Configuration:
    - `extra = "ignore"`: This configuration setting ensures that any additional fields not defined in this model
                          are ignored during deserialization. This helps in maintaining backward compatibility
                          and avoiding errors if new fields are added to the underlying data structure but not yet
                          accounted for in this model.

    Validator:
    - `verify_char_limit`: A model validator that checks if the length of the `value` attribute exceeds the `limit`.
                           If it does, a `ValueError` is raised with an appropriate error message indicating the
                           character limit violation and details about the block.

    Overridden Method:
    - `__setattr__`: This method is overridden to perform additional validation whenever the `value` attribute is set.
                    After setting the attribute, it validates the entire model using `model_validate` based on the
                    current state of the object (excluding unset fields). This helps in maintaining data integrity
                    and ensuring that the block remains valid after any changes to its content.
    """
    __id_prefix__ = "block"
    value: str = Field(..., description="Value of the block.")
    limit: int = Field(CORE_MEMORY_BLOCK_CHAR_LIMIT, description="Character limit of the block.")
    template_name: Optional[str] = Field(None, description="Name of the block if it is a template.", alias="name")
    is_template: bool = Field(False, description="Whether the block is a template (e.g. saved human/persona options).")
    label: Optional[str] = Field(None, description="Label of the block (e.g. 'human', 'persona') in the context window.")
    description: Optional[str] = Field(None, description="Description of the block.")
    metadata_: Optional[dict] = Field({}, description="Metadata of the block.")

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def verify_char_limit(self) -> Self:
        if self.value and len(self.value) > self.limit:
            error_msg = f"Edit failed: Exceeds {self.limit} character limit (requested {len(self.value)}) - {str(self)}."
            raise ValueError(error_msg)
        return self

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "value":
            self.__class__.model_validate(self.model_dump(exclude_unset=True))


class Block(BaseBlock):
    """
    Extends the `BaseBlock` class to add specific identifiers related to the organization, creator, and last updater
    of the block.

    Attributes:
    - `id`: A unique identifier for the block, generated using the `generate_id_field` method inherited from
            `BaseBlock` or its superclass.
    - `organization_id`: An optional unique identifier for the organization associated with the block. This can be
                         used to group blocks under specific organizations in a multi-tenant or hierarchical setup.
    - `created_by_id`: An optional identifier for the user who created the block. This helps in tracking the origin
                       of the block within the system.
    - `last_updated_by_id`: An optional identifier for the user who last updated the block. Useful for auditing and
                            keeping track of changes made to the block over time.
    """
    id: str = BaseBlock.generate_id_field()
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the block.")
    created_by_id: Optional[str] = Field(None, description="The id of the user that made this Block.")
    last_updated_by_id: Optional[str] = Field(None, description="The id of the user that last updated this Block.")


class Human(Block):
    """
    Represents a specific type of block labeled as "human". It inherits from the `Block` class and sets the `label`
    attribute to "human" by default. This likely indicates that the block's content is related to a human role or
    input in a specific context, such as a chat conversation or a role-playing scenario.
    """
    label: str = "human"


class Persona(Block):
    """
    Similar to the `Human` class, this represents a specific type of block labeled as "persona". It inherits from
    `Block` and sets the `label` to "persona" by default. This suggests that the block's content is associated with
    a specific persona or character role in a given context, perhaps used for defining different personalities in
    an interaction.
    """
    label: str = "persona"


class BlockLabelUpdate(BaseModel):
    """
    Represents the data needed to update the label of a block.

    Attributes:
    - `current_label`: The current label of the block. This is a required field and is used to identify the block
                       that needs its label updated.
    - `new_label`: The new label that will be assigned to the block. This is also a required field and defines the
                   updated label value.
    """
    current_label: str = Field(..., description="Current label of the block.")
    new_label: str = Field(..., description="New label of the block.")


class BlockUpdate(BaseBlock):
    """
    Represents the data that can be updated for an existing block. It inherits from `BaseBlock` and allows for
    optional updates to the `limit` and `value` attributes.

    Attributes:
    - `limit`: An optional integer representing the updated character limit of the block. By default, it uses the
               same value as `CORE_MEMORY_BLOCK_CHAR_LIMIT` if not specified.
    - `value`: An optional string representing the updated value of the block. This allows for modifying the main
               content of the block.

    Class Configuration:
    - `extra = "ignore"`: Inherits the same configuration as `BaseBlock` to ignore additional fields during
                          deserialization for backward compatibility.
    """
    limit: Optional[int] = Field(CORE_MEMORY_BLOCK_CHAR_LIMIT, description="Character limit of the block.")
    value: Optional[str] = Field(None, description="Value of the block.")

    class Config:
        extra = "ignore"


class BlockLimitUpdate(BaseModel):
    """
    Represents the data needed to update the character limit of a block.

    Attributes:
    - `label`: The label of the block. This is a required field and is used to identify the specific block for which
               the limit needs to be updated.
    - `limit`: The new character limit that will be set for the block. This is also a required field and defines the
               updated limit value.
    """
    label: str = Field(..., description="Label of the block.")
    limit: int = Field(..., description="New limit of the block.")


class CreateBlock(BaseBlock):
    """
    Represents the data needed to create a new block. It inherits from `BaseBlock` and requires specific values for
    the `label` and `value` attributes, while also setting a default value for the `is_template` attribute to
    `False`.

    Attributes:
    - `label`: The label of the block. This is a required field and is used to categorize or identify the block in a
               specific context, similar to other block classes.
    - `limit`: An integer representing the character limit of the block. By default, it uses the value from
               `CORE_MEMORY_BLOCK_CHAR_LIMIT`.
    - `value`: The main text content of the block. This is a required field and holds the actual data for the new
               block.
    - `is_template`: A boolean indicating whether the block is a template. By default, it's set to `False`, meaning
                     the block is not created as a template.
    - `template_name`: An optional string representing the name of the block if it is a template. Similar to other
                       related attributes, it has an alias "name" for serialization/deserialization purposes.
    """
    label: str = Field(..., description="Label of the block.")
    limit: int = Field(CORE_MEMORY_BLOCK_CHAR_LIMIT, description="Character limit of the block.")
    value: str = Field(..., description="Value of the block.")
    is_template: bool = False
    template_name: Optional[str] = Field(None, description="Name of the block if it is a template.", alias="name")


class CreateHuman(CreateBlock):
    """
    Represents the data needed to create a new "human" type of block. It inherits from `CreateBlock` and sets the
    `label` attribute to "human" by default, indicating that the new block will be related to a human role or input.
    """
    label: str = "human"


class CreatePersona(CreateBlock):
    """
    Similar to `CreateHuman`, this represents the data needed to create a new "persona" type of block. It inherits
    from `CreateBlock` and sets the `label` attribute to "persona" by default, signifying that the new block will
    be associated with a specific persona or character role.
    """
    label: str = "persona"


class CreateBlockTemplate(CreateBlock):
    """
    Represents the data needed to create a new block that is specifically a template. It inherits from `CreateBlock`
    and sets the `is_template` attribute to `True` by default, indicating that the new block will be created as a
    reusable template.
    """
    is_template: bool = True


class CreateHumanBlockTemplate(CreateHuman):
    """
    Represents the data needed to create a new "human" type of block that is also a template. It inherits from
    `CreateHuman` and sets the `is_template` attribute to `True` while keeping the `label` as "human", allowing for
    creating a reusable "human" block template.
    """
    is_template: bool = True
    label: str = "human"


class CreatePersonaBlockTemplate(CreatePersona):
    """
    Similar to `CreateHumanBlockTemplate`, this represents the data needed to create a new "persona" type of block
    that is a template. It inherits from `CreatePersona` and sets the `is_template` attribute to `True` while
    maintaining the `label` as "persona", enabling the creation of a reusable "persona" block template.
    """
    is_template: bool = True
    label: str = "persona"