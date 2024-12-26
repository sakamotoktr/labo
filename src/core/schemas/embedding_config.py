from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """
    Represents the configuration for an embedding model within a system.

    This class defines various attributes that specify how an embedding model should be accessed and configured,
    including details about the endpoint type, the actual model to use, its dimensionality, and other related
    parameters.

    Attributes:
    - `embedding_endpoint_type`: A literal value specifying the type of the endpoint where the embedding model is
                                 hosted or accessed. It can take one of several predefined values such as "openai",
                                 "anthropic", etc., indicating different providers or platforms. This is a required
                                 field as it determines the overall source or technology used for the embeddings.
    - `embedding_endpoint`: An optional string representing the actual endpoint URL for the model. If it's `None`,
                            it likely indicates that the model is running locally. This allows for flexibility in
                            configuring whether to use a remote or local embedding service.
    - `embedding_model`: A required string representing the name of the specific embedding model to be used. This
                         identifies which particular model implementation will generate the embeddings.
    - `embedding_dim`: An integer representing the dimension of the embedding vectors produced by the model. This is
                       a required field as it defines the size and structure of the resulting embeddings.
    - `embedding_chunk_size`: An optional integer representing the chunk size used for processing data when creating
                              embeddings. It defaults to 300, and this setting can impact how data is divided and
                              processed during the embedding generation process.
    - `handle`: An optional string representing a handle for this configuration in the format "provider/model-name".
                This can be used for identification or referencing the configuration in a more user-friendly or
                standardized way.
    - `azure_endpoint`: An optional string representing the Azure endpoint for the model. This is relevant only if
                        the `embedding_endpoint_type` is set to "azure" and provides the specific URL for the Azure
                        service being used.
    - `azure_version`: An optional string representing the Azure version for the model. Similar to the `azure_endpoint`,
                       this is specific to Azure deployments and can be used to specify the version of the service.
    - `azure_deployment`: An optional string representing the Azure deployment for the model. It helps in identifying
                          the specific deployment instance within the Azure environment.

    Class Methods:
    - `default_config`: A class method that returns a default `EmbeddingConfig` instance based on the provided
                        `model_name` and `provider` parameters. It has predefined configurations for specific
                        models like "text-embedding-ada-002" (for OpenAI) and "labo" (using a Hugging Face model),
                        and raises a `ValueError` if an unsupported model name is provided.

    Methods:
    - `pretty_print`: A method that returns a formatted string representation of the `EmbeddingConfig` instance,
                       showing the model name along with optional details like the endpoint type and IP (endpoint
                       URL), if available. This can be useful for presenting the configuration in a human-readable
                       format.
    """
    embedding_endpoint_type: Literal[
        "openai",
        "anthropic",
        "cohere",
        "google_ai",
        "azure",
        "groq",
        "ollama",
        "webui",
        "webui-legacy",
        "lmstudio",
        "lmstudio-legacy",
        "llamacpp",
        "koboldcpp",
        "vllm",
        "hugging-face",
        "mistral",
        "together",
    ] = Field(..., description="The endpoint type for the model.")
    embedding_endpoint: Optional[str] = Field(None, description="The endpoint for the model (`None` if local).")
    embedding_model: str = Field(..., description="The model for the embedding.")
    embedding_dim: int = Field(..., description="The dimension of the embedding.")
    embedding_chunk_size: Optional[int] = Field(300, description="The chunk size of the embedding.")
    handle: Optional[str] = Field(None, description="The handle for this config, in the format provider/model-name.")
    azure_endpoint: Optional[str] = Field(None, description="The Azure endpoint for the model.")
    azure_version: Optional[str] = Field(None, description="The Azure version for the model.")
    azure_deployment: Optional[str] = Field(None, description="The Azure deployment for the model.")

    @classmethod
    def default_config(cls, model_name: Optional[str] = None, provider: Optional[str] = None):
        """
        Generate a default `EmbeddingConfig` instance based on the provided model name and provider.

        This method checks the given `model_name` and `provider` values and returns a predefined configuration
        for specific known models. If the `model_name` is "text-embedding-ada-002" or the provider is "openai"
        (with no specific model name provided), it returns a configuration for the OpenAI text-embedding-ada-002
        model. If the `model_name` is "labo", it returns a configuration using a specific Hugging Face model.
        For any other unsupported model name, it raises a `ValueError`.

        Args:
        - `model_name`: An optional string representing the name of the model. This is used to determine which
                        default configuration to return.
        - `provider`: An optional string representing the provider of the model. This can also be used in
                      combination with `model_name` to identify the appropriate default configuration.

        Returns:
        - `EmbeddingConfig`: A default `EmbeddingConfig` instance with predefined settings for the identified
                             model or provider.

        Raises:
        - `ValueError`: If the provided `model_name` corresponds to an unsupported model.
        """
        if model_name == "text-embedding-ada-002" or (not model_name and provider == "openai"):
            return cls(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
            )
        elif model_name == "labo":
            return cls(
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
                embedding_chunk_size=300,
                embedding_endpoint_type="hugging-face",
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def pretty_print(self) -> str:
        """
        Generate a human-readable string representation of the `EmbeddingConfig` instance.

        This method constructs a string that includes the `embedding_model` name and, optionally, additional
        details like the `embedding_endpoint_type` and `embedding_endpoint` if they are set. This provides a
        convenient way to display the key aspects of the embedding configuration in a formatted manner.

        Returns:
        - `str`: A formatted string representing the `EmbeddingConfig` instance.
        """
        return (
            f"{self.embedding_model}"
            + (f" [type={self.embedding_endpoint_type}]" if self.embedding_endpoint_type else "")
            + (f" [ip={self.embedding_endpoint}]" if self.embedding_endpoint else "")
        )