import uuid
from typing import Any, List, Optional

import numpy as np
import tiktoken


def segment_content(content: str, segment_length: int) -> List[str]:
    from llama_index.core import Document as TextDoc
    from llama_index.core.node_parser import SentenceSplitter

    content_divider = SentenceSplitter(chunk_size=segment_length)
    source_docs = [TextDoc(text=content)]
    fragments = content_divider.get_nodes_from_documents(source_docs)
    return [element.text for element in fragments]


def limit_content_length(content: str, limit: int, tokenizer) -> str:
    encoded_content = tokenizer.encode(content)[:limit]
    return tokenizer.decode(encoded_content)


def validate_and_partition_content(content: str, vector_engine: str) -> List[str]:
    """Divide content into fragments that respect token limitations"""

    if vector_engine in TOKEN_ENGINE_MAPPING:
        tokenizer = tiktoken.get_encoding(TOKEN_ENGINE_MAPPING[vector_engine])
    else:
        print(
            f"Alert: No tokenizer found for {vector_engine}, defaulting to {DEFAULT_TOKEN_ENGINE}"
        )
        tokenizer = tiktoken.get_encoding(DEFAULT_TOKEN_ENGINE)

    token_count = len(tokenizer.encode(content))

    if hasattr(tokenizer, "limit"):
        max_tokens = tokenizer.limit
    else:
        print(
            f"Alert: No token limit found for {vector_engine}, using fallback limit of 8191"
        )
        max_tokens = 8191

    if token_count > max_tokens:
        print(
            f"Alert: Content exceeds {max_tokens} tokens ({token_count} found). Trimming content."
        )
        processed_content = preprocess_content(content, vector_engine)
        content = limit_content_length(processed_content, max_tokens, tokenizer)

    return [content]


class VectorEndpoint:
    def __init__(
        self,
        engine_type: str,
        service_url: str,
        client_id: str,
        request_timeout: float = 60.0,
        **params: Any,
    ):
        if not verify_url(service_url):
            raise ValueError(
                f"Invalid vector service URL provided: '{service_url}'. Verify configuration settings."
            )
        if engine_type == "basic-service":
            engine_type = "advanced-model-v1.5"
        self.engine_id = engine_type
        self._client = client_id
        self._endpoint = service_url
        self._wait_time = request_timeout

    def _request_vector(self, content: str) -> List[float]:
        if not verify_url(self._endpoint):
            raise ValueError(
                f"Invalid vector service URL configured: '{self._endpoint}'. Check configuration."
            )
        import httpx

        headers = {"Content-Type": "application/json"}
        payload = {"input": content, "model": self.engine_id, "user": self._client}

        with httpx.Client() as session:
            response = session.post(
                f"{self._endpoint}/embeddings",
                headers=headers,
                json=payload,
                timeout=self._wait_time,
            )

        result = response.json()

        if isinstance(result, list):
            vector_data = result
        elif isinstance(result, dict):
            try:
                vector_data = result["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(f"Unexpected vector service response format: {result}")
        else:
            raise TypeError(f"Unrecognized vector service response: {result}")

        return vector_data

    def calculate_vector(self, content: str) -> List[float]:
        return self._request_vector(content)


class CloudVectorService:
    def __init__(self, service_url: str, auth_key: str, api_ver: str, engine_type: str):
        from openai import AzureOpenAI

        self.engine = AzureOpenAI(
            api_key=auth_key, api_version=api_ver, azure_endpoint=service_url
        )
        self.engine_type = engine_type

    def calculate_vector(self, content: str):
        vector = (
            self.engine.embeddings.create(input=[content], model=self.engine_type)
            .data[0]
            .embedding
        )
        return vector


class LocalVectorService:
    def __init__(self, engine_type: str, service_url: str, extra_params: dict):
        self.engine_type = engine_type
        self.service_url = service_url
        self.extra_params = extra_params

    def calculate_vector(self, content: str):
        import httpx

        headers = {"Content-Type": "application/json"}
        payload = {"model": self.engine_type, "prompt": content}
        payload.update(self.extra_params)

        with httpx.Client() as session:
            response = session.post(
                f"{self.service_url}/api/embeddings",
                headers=headers,
                json=payload,
            )

        result = response.json()
        return result["embedding"]


def prepare_search_vector(vector_service, search_text: str):
    """Create padded vector for database queries"""
    vector = vector_service.calculate_vector(search_text)
    vector_array = np.array(vector)
    padded_vector = np.pad(
        vector_array, (0, VECTOR_MAX_DIM - vector_array.shape[0]), mode="constant"
    ).tolist()
    return padded_vector


def create_vector_service(config: VectorConfig, client_id: Optional[uuid.UUID] = None):
    """Initialize appropriate vector service based on configuration"""

    service_type = config.vector_service_type

    if service_type == "standard":
        from llama_index.embeddings.openai import OpenAIEmbedding

        extra_params = {"user_id": client_id} if client_id else {}
        service = OpenAIEmbedding(
            api_base=config.vector_endpoint,
            api_key=model_settings.openai_api_key,
            additional_kwargs=extra_params,
        )
        return service

    elif service_type == "cloud":
        required_settings = [
            model_settings.azure_api_key,
            model_settings.azure_base_url,
            model_settings.azure_api_version,
        ]
        if not all(required_settings):
            raise ValueError("Missing required cloud service credentials")

        return CloudVectorService(
            service_url=model_settings.azure_base_url,
            auth_key=model_settings.azure_api_key,
            api_ver=model_settings.azure_api_version,
            engine_type=config.vector_model,
        )

    elif service_type == "custom":
        return VectorEndpoint(
            engine_type=config.vector_model,
            service_url=config.vector_endpoint,
            client_id=client_id,
        )
    elif service_type == "local":
        return LocalVectorService(
            engine_type=config.vector_model,
            service_url=config.vector_endpoint,
            extra_params={},
        )
    else:
        raise ValueError(f"Unsupported vector service type: {service_type}")
