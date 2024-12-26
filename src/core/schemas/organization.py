from datetime import datetime
from typing import Optional

from pydantic import Field

from labo.schemas.labo_base import LABOBase
from labo.utils import create_random_username, get_utc_time


class OrganizationBase(LABOBase):
    """
    Represents the base structure for organizations in the system.

    This class inherits from `LABOBase` and is likely used to provide common functionality or attributes for
    more specific organization-related classes. It defines a prefix (`"org"`) for the unique identifiers of
    organizations, which helps in creating consistent and identifiable IDs for organizations within the system.

    Attributes:
    - `__id_prefix__`: A class attribute set to `"org"`, which is used to prefix the unique identifiers of
                       organizations. This is likely used in the process of generating unique IDs for each
                       organization instance.
    """
    __id_prefix__ = "org"