from datetime import datetime
from typing import Optional

from pydantic import Field

from labo.schemas.enums import JobStatus
from labo.schemas.labo_base import OrmMetadataBase


class JobBase(OrmMetadataBase):
    """
    Represents the base structure for a job in the system, providing common attributes that are relevant to all jobs.

    This class inherits from `OrmMetadataBase` which might provide additional metadata-related functionality or
    attributes related to object-relational mapping.

    Attributes:
    - `__id_prefix__`: A prefix used for generating the job's ID. Set to "job", it helps in identifying and categorizing
                       the job within a larger system where IDs follow a specific naming convention.
    - `status`: An instance of the `JobStatus` enumeration representing the current status of the job. By default,
                it's set to `JobStatus.created`, indicating that the job has just been created and not yet started
                or completed. This attribute is crucial for tracking the progress of the job throughout its lifecycle.
    - `completed_at`: An optional `datetime` object representing the Unix timestamp of when the job was completed.
                      If the job is still in progress or not yet completed, this value will be `None`. It's used to
                      record the exact time when the job finishes successfully.
    - `metadata_`: An optional dictionary containing additional metadata related to the job. This could include
                   custom tags, user-defined settings, or other information relevant to the specific job. Note the
                   alias "metadata_", which might be used for serialization/deserialization purposes.
    """
    __id_prefix__ = "job"
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    metadata_: Optional[dict] = Field(None, description="The metadata of the job.")


class Job(JobBase):
    """
    Extends the `JobBase` class to add an identifier for the user associated with the job and a unique ID for the job itself.

    Attributes:
    - `id`: A unique identifier for the job, generated using the `generate_id_field` method inherited from `JobBase`
            or its superclass. This ID is used to distinguish different jobs in the system.
    - `user_id`: An optional unique identifier for the user who initiated or is associated with the job. This can be
                 useful for tracking which user is responsible for a particular job, perhaps for auditing,
                 permission management, or reporting purposes.
    """
    id: str = JobBase.generate_id_field()
    user_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the job.")


class JobUpdate(JobBase):
    """
    Represents the data that can be used to update an existing job's information.

    It inherits from `JobBase` and allows for the optional update of the job's status.

    Attributes:
    - `status`: An optional instance of the `JobStatus` enumeration representing the updated status of the job.
                This allows changing the job's status during its execution, for example, from `JobStatus.created`
                to `JobStatus.running` or `JobStatus.completed`.

    Class Configuration:
    - `extra = "ignore"`: This configuration setting ensures that any additional fields not defined in this model
                          are ignored during deserialization. This helps in maintaining backward compatibility
                          and avoiding errors if new fields are added to the underlying data structure but not yet
                          accounted for in this model.
    """
    status: Optional[JobStatus] = Field(None, description="The status of the job.")

    class Config:
        extra = "ignore"