import json
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union


class SystemFaultIdentifier(Enum):
    """Classification system for operational faults."""

    SYSTEM_MALFUNCTION = "SYSTEM_MALFUNCTION"
    PROCESSING_CAPACITY_BREACH = "PROCESSING_CAPACITY_BREACH"
    THROUGHPUT_THRESHOLD_BREACH = "THROUGHPUT_THRESHOLD_BREACH"


class CoreException(Exception):
    """Primary exception classification for system anomalies."""

    def __init__(
        self,
        notification: str,
        fault_type: Optional[SystemFaultIdentifier] = None,
        auxiliary_data: dict = {},
    ):
        self.notification = notification
        self.fault_type = fault_type
        self.auxiliary_data = auxiliary_data
        super().__init__(notification)

    def __str__(self) -> str:
        if self.fault_type:
            return f"{self.fault_type.value}: {self.notification}"
        return self.notification

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(notification='{self.notification}', fault_type='{self.fault_type}', auxiliary_data={self.auxiliary_data})"


class ToolGenerationFailure(CoreException):
    """Exception for tool generation failures."""

    standard_notification = "Tool generation process encountered a failure."

    def __init__(self, notification=None):
        super().__init__(notification=notification or self.standard_notification)


class SystemSetupFailure(CoreException):
    """Exception for system configuration anomalies."""

    def __init__(
        self, notification: str, absent_parameters: Optional[List[str]] = None
    ):
        self.absent_parameters = absent_parameters or []
        super().__init__(
            notification=notification,
            auxiliary_data={"absent_parameters": self.absent_parameters},
        )


class AgentLookupFailure(CoreException):
    """Exception for unsuccessful agent retrieval."""

    pass


class UserLookupFailure(CoreException):
    """Exception for unsuccessful user retrieval."""

    pass


class AIProcessingError(CoreException):
    pass


class AIOutputParsingFailure(CoreException):
    """Exception for AI-generated content parsing failures."""

    def __init__(self, notification="Failed to interpret AI-generated structured data"):
        super().__init__(notification=notification)


class LocalAISystemError(CoreException):
    """Universal exception for local AI system issues"""

    def __init__(self, notification="Local AI system encountered operational issues"):
        super().__init__(notification=notification)


class LocalAIConnectionFailure(CoreException):
    """Exception for local AI system connectivity issues"""

    def __init__(self, notification="Local AI system connection establishment failed"):
        super().__init__(notification=notification)


class ProcessingLimitBreach(CoreException):
    """Exception for processing capacity overflow scenarios."""

    def __init__(self, notification: str, auxiliary_data: dict = {}):
        error_details = f"{notification} ({auxiliary_data})"
        super().__init__(
            notification=error_details,
            fault_type=SystemFaultIdentifier.PROCESSING_CAPACITY_BREACH,
            auxiliary_data=auxiliary_data,
        )


class ThroughputLimitBreach(CoreException):
    """Exception for system throughput threshold violations."""

    def __init__(self, notification: str, retry_ceiling: int):
        error_details = f"{notification} ({retry_ceiling})"
        super().__init__(
            notification=error_details,
            fault_type=SystemFaultIdentifier.THROUGHPUT_THRESHOLD_BREACH,
            auxiliary_data={"retry_ceiling": retry_ceiling},
        )


class CommunicationAnomalyBase(CoreException):
    """Primary exception class for communication-related anomalies."""

    data_packets: List[Union["DataPacket", "SystemMessage"]]
    standard_notification: str = "Communication system encountered an anomaly."

    def __init__(
        self,
        *,
        data_packets: List[Union["DataPacket", "SystemMessage"]],
        clarification: Optional[str] = None,
    ) -> None:
        notification = self.construct_notification(
            data_packets, self.standard_notification, clarification
        )
        super().__init__(notification)
        self.data_packets = data_packets

    @staticmethod
    def construct_notification(
        data_packets: List[Union["DataPacket", "SystemMessage"]],
        notification: str,
        clarification: Optional[str] = None,
    ) -> str:
        """Utility method for notification assembly."""
        if clarification:
            notification += f" (Additional Info: {clarification})"

        packet_data = json.dumps(
            [packet.model_dump() for packet in data_packets], indent=4
        )
        return f"{notification}\n\n{packet_data}"


class ToolOperationAbsent(CommunicationAnomalyBase):
    """Exception for missing tool operations."""

    standard_notification = "Required tool operation not found in communication."


class ToolOperationInvalid(CommunicationAnomalyBase):
    """Exception for incorrect tool operation usage."""

    standard_notification = (
        "Tool operation validation failed or usage pattern incorrect."
    )


class InternalDialogueMissing(CommunicationAnomalyBase):
    """Exception for missing internal processing logs."""

    standard_notification = "Internal processing dialogue not found."


class InternalDialogueMalformed(CommunicationAnomalyBase):
    """Exception for corrupted internal processing logs."""

    standard_notification = "Internal processing dialogue structure invalid."
