from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json

# Global variables
request_handler: Any = None
api_resources: Any = None
completion_config: Any = None
object_types: Dict[str, Any] = {}


class ServiceType(Enum):
    AZURE = 1
    OPENAI = 2
    AZURE_AD = 3

    @staticmethod
    def from_string(label: str) -> "ServiceType":
        mapping = {
            "azure": ServiceType.AZURE,
            "azure_ad": ServiceType.AZURE_AD,
            "azuread": ServiceType.AZURE_AD,
            "open_ai": ServiceType.OPENAI,
            "openai": ServiceType.OPENAI,
        }
        if label.lower() not in mapping:
            raise ValueError(
                "Invalid service type. Choose from: 'azure', 'azure_ad', 'open_ai'"
            )
        return mapping[label.lower()]


class ServiceResponse:
    def __init__(self, headers: Dict[str, str], data: Any):
        self.headers = headers
        self.data = data

    @property
    def request_id(self) -> str:
        return self.headers.get("request-id", "")

    @property
    def organization(self) -> str:
        return self.headers.get("OpenAI-Organization", "")

    @property
    def processing_ms(self) -> Optional[int]:
        if ms := self.headers.get("Openai-Processing-Ms"):
            return int(ms)
        return None


class ServiceObject:
    def __init__(self, identifier: str = "", **kwargs):
        self._data: Dict[str, Any] = {}
        if identifier:
            self._data["id"] = identifier

        self.auth_key: str = kwargs.get("auth_key", "")
        self.api_version: str = kwargs.get("api_version", "")
        self.service_type: ServiceType = kwargs.get("service_type", ServiceType.OPENAI)
        self.org: str = kwargs.get("org", "")
        self.response_time: Optional[int] = kwargs.get("response_time")
        self.api_base_override: str = kwargs.get("api_base_override", "")
        self.model: str = kwargs.get("model", "")
        self.retrieve_params: Dict[str, Any] = kwargs.get("retrieve_params", {})

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        if value == "":
            raise ValueError(
                f"Cannot set {key} to empty string - use None to remove property"
            )
        self._data[key] = value

    def delete(self, key: str) -> None:
        raise NotImplementedError("Delete operation not supported")

    def to_json(self) -> str:
        return json.dumps(self._data, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def to_dict_recursive(self) -> Dict[str, Any]:
        result = {}
        for k, v in self._data.items():
            if isinstance(v, ServiceObject):
                result[k] = v.to_dict_recursive()
            elif isinstance(v, list):
                result[k] = [
                    (
                        item.to_dict_recursive()
                        if isinstance(item, ServiceObject)
                        else item
                    )
                    for item in v
                ]
            else:
                result[k] = v
        return result

    @property
    def api_base(self) -> str:
        return ""

    async def make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
        raw_response: bool = False,
        request_id: str = "",
        timeout: float = 0.0,
    ) -> Any:
        params = params or self.retrieve_params

        requester = RequestHandler(
            self.auth_key,
            self.api_base_override,
            self.api_base,
            self.service_type,
            self.api_version,
            self.org,
        )

        response, stream_result, auth_key = await requester.send_request(
            method, url, params, stream, headers, request_id, timeout
        )

        if stream:
            if isinstance(response, ServiceResponse):
                raise ValueError("Expected stream response, got ServiceResponse")

            async def process_stream():
                async for line in stream_result:
                    converted = convert_to_service_object(
                        line,
                        auth_key=auth_key,
                        api_version=self.api_version,
                        org=self.org,
                    )
                    yield line if raw_response else converted

            return process_stream()

        return convert_to_service_object(
            response, auth_key=auth_key, api_version=self.api_version, org=self.org
        )


class RequestHandler:
    def __init__(
        self,
        auth_key: str,
        api_base_override: str,
        api_base: str,
        service_type: ServiceType,
        api_version: str,
        org: str,
    ):
        self.auth_key = auth_key
        self.api_base = api_base_override or api_base
        self.service_type = service_type
        self.api_version = api_version
        self.org = org

    async def send_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        stream: bool,
        headers: Optional[Dict[str, str]],
        request_id: str,
        timeout: float,
    ) -> tuple[Any, Any, str]:
        # Implementation would go here
        # This would handle the actual HTTP requests
        raise NotImplementedError("Request handling not implemented")


def convert_to_service_object(response: Any, **kwargs) -> Any:
    if isinstance(response, ServiceResponse):
        kwargs["org"] = response.organization
        if ms := response.processing_ms:
            kwargs["response_time"] = ms
        return convert_to_service_object(response.data, **kwargs)

    if isinstance(response, list):
        return [convert_to_service_object(item, **kwargs) for item in response]

    if isinstance(response, dict):
        if obj_type := response.get("object"):
            if cls := object_types.get(obj_type):
                return cls

        obj = ServiceObject("", **kwargs)
        for k, v in response.items():
            obj.set(k, convert_to_service_object(v, **kwargs))
        return obj

    return response
