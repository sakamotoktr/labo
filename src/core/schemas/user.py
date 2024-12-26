from datetime import datetime
from typing import Optional

from pydantic import Field

from labo.schemas.labo_base import LABOBase
from labo.services.organization_manager import OrganizationManager


class UserBase(LABOBase):
    """
    Represents the base structure for users in the system.

    This class inherits from `LABOBase` and is likely used to establish a common foundation for more specific
    user-related classes. It defines a prefix (`"user"`) for the unique identifiers of users, which helps in
    creating consistent and identifiable IDs for different users within the system.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"user"`, which is used to prefix the unique identifiers of
                       users. This is likely used in the process of generating unique IDs for each user
                       instance.
    """
    __id_prefix__ = "user"

class User(UserBase):
    """
    Represents a user within the system, including their identifier, organization association, name, and status.

    This class inherits from `UserBase` and adds specific attributes that define a user in more detail. It
    includes a unique ID generated using the `generate_id_field` method from the base class, an organization
    ID that links the user to a particular organization (using a default value if not provided explicitly), a
    required name that represents the user's identity, creation and update dates that are automatically set
    to the current UTC time when the user object is created or updated, and a boolean flag indicating whether
    the user has been marked as deleted.

    Attributes:
    - `id`: A unique identifier for the user, generated using the mechanism provided by the base class. This ID
            is used to distinguish one user from another within the system.
    - `organization_id`: An optional string representing the ID of the organization to which the user belongs.
                         By default, it's set to the value from `OrganizationManager.DEFAULT_ORG_ID`, which
                         likely provides a default organization for users if not specified otherwise.
    - `name`: A required string representing the name of the user. This is a key identifying attribute for
              the user.
    - `created_at`: An optional `datetime` object representing the date and time when the user was created. By
                    default, it's set to the current UTC time using `datetime.utcnow` as the default factory.
    - `updated_at`: An optional `datetime` object representing the date and time when the user was last updated.
                    Similar to `created_at`, it's set to the current UTC time by default.
    - `is_deleted`: A boolean indicating whether the user has been marked as deleted. By default, it's set to
                    `False`, meaning the user is considered active or not deleted.
    """
    id: str = UserBase.generate_id_field()
    organization_id: Optional[str] = Field(OrganizationManager.DEFAULT_ORG_ID, description="The organization id of the user")
    name: str = Field(..., description="The name of the user.")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The creation date of the user.")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="The update date of the user.")
    is_deleted: bool = Field(False, description="Whether this user is deleted or not.")