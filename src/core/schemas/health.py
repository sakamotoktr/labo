from pydantic import BaseModel


class Health(BaseModel):
    """
    The `Health` class is a Pydantic model that is likely used to represent the health status of a system or a component within a system.
    It serves as a standardized way to communicate information about the version of the system and its current operational status.
    Attributes:
    - `version`: A string representing the version number or identifier of the system. This can be used to track which specific release or build of the system is currently running.
                  For example, it could be in a format like "1.0.0", "v2.1", etc., depending on the versioning scheme used by the developers.
    - `status`: A string indicating the current status of the system. Common values for this could be "healthy", "unhealthy", "degraded", etc., to convey whether the system is functioning properly,
                experiencing issues, or operating with reduced functionality. This allows other parts of the system or external monitoring tools to quickly assess the overall well-being of the component.
    """
    version: str
    status: str