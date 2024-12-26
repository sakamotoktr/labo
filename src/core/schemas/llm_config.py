from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, root_validator


class LLMConfig(BaseModel):
    """
    Represents the configuration for a language model (LLM) within a system.

    This class defines the necessary parameters and settings required to interact with an LLM, including details
    about the model itself, its endpoint, and other related options.

    Attributes:
    - `model`: A required string representing the name of the LLM model. This is used to identify which specific
              model implementation will be used for text generation or other tasks.
    - `model_endpoint_type`: A literal value specifying the type of the endpoint where the LLM model is hosted or
                             accessed. It can take one of several predefined values such as "openai", "anthropic",
                             etc., indicating different providers or platforms. This is a required field as it
                             determines the overall source or technology used for the model.
    - `model_endpoint`: An optional string representing the actual endpoint URL for the model. If it's `None`,
                        it might imply that the model is running locally or that the endpoint needs to be
                        configured separately. This allows for flexibility in setting up the model access.
    - `model_wrapper`: An optional string representing the wrapper used for the model. A wrapper could be a layer
                       that provides additional functionality or compatibility with the system, such as handling
                       input/output formatting or integrating with specific frameworks.
    - `context_window`: An integer representing the size of the context window for the model. The context window
                         defines the amount of text (usually in terms of tokens) that the model can consider at
                         once during text generation or processing. This is a required field as it impacts the
                         model's ability to handle long sequences of text.
    - `put_inner_thoughts_in_kwargs`: An optional boolean indicating whether to include "inner_thoughts" as a
                                      keyword argument in the function call. If set to `True`, it can help with
                                      function calling performance and the generation of inner thoughts. By default,
                                      it's set to `True`, but there are specific models like "gpt-4" for which it can
                                      be overridden.
    - `handle`: An optional string representing a handle for this configuration in the format "provider/model-name".
                This can be used for identification or referencing the configuration in a more user-friendly or
                standardized way.

    Class Configuration:
    - `model_config`: A `ConfigDict` object with `protected_namespaces` set to an empty tuple. This configuration
                      setting might be related to protecting certain namespaces within the model's attributes or
                      behavior, although its exact usage would depend on the broader context of the application.

    Validators:
    - `set_default_put_inner_thoughts`: A root validator that runs before other validations. It checks if the
                                        `put_inner_thoughts_in_kwargs` value is `None` and, based on the model name,
                                        sets it to an appropriate default value. For models like "gpt-4", it sets it to
                                        `False`, while for others, it sets it to `True`.

    Class Methods:
    - `default_config`: A class method that returns a default `LLMConfig` instance based on the provided `model_name`.
                        It has predefined configurations for specific well-known models like "gpt-4", "gpt-4o-mini",
                        and "labo". If an unsupported model name is provided, it raises a `ValueError`.

    Methods:
    - `pretty_print`: A method that generates a human-readable string representation of the `LLMConfig` instance.
                       It includes the model name along with optional details like the endpoint type and IP (endpoint
                       URL), if available. This can be useful for presenting the configuration in a clear and
                       understandable format.
    """
    model: str = Field(..., description="LLM model name. ")
    model_endpoint_type: Literal[
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
    model_endpoint: Optional[str] = Field(None, description="The endpoint for the model.")
    model_wrapper: Optional[str] = Field(None, description="The wrapper for the model.")
    context_window: int = Field(..., description="The context window size for the model.")
    put_inner_thoughts_in_kwargs: Optional[bool] = Field(
        True,
        description="Puts 'inner_thoughts' as a kwarg in the function call if this is set to True. This helps with function calling performance and also the generation of inner thoughts.",
    )
    handle: Optional[str] = Field(None, description="The handle for this config, in the format provider/model-name.")

    model_config = ConfigDict(protected_namespaces=())

    @root_validator(pre=True)
    def set_default_put_inner_thoughts(cls, values):
        """
        Set the default value for `put_inner_thoughts_in_kwargs` based on the model name.

        This root validator runs before other validations. It checks if the `put_inner_thoughts_in_kwargs` value
        is `None` in the provided `values` dictionary. If it is, it determines an appropriate default value
        based on the model name. For models in the `avoid_put_inner_thoughts_in_kwargs` list (currently only
        "gpt-4"), it sets the value to `False`; otherwise, it sets it to `True`.

        Args:
        - `cls`: The class itself (a reference to `LLMConfig`).
        - `values`: A dictionary containing the values of the model's attributes before validation.

        Returns:
        - `dict`: The updated `values` dictionary with the `put_inner_thoughts_in_kwargs` value set appropriately.
        """
        model = values.get("model")
        avoid_put_inner_thoughts_in_kwargs = ["gpt-4"]
        if values.get("put_inner_thoughts_in_kwargs") is None:
            values["put_inner_thoughts_in_kwargs"] = False if model in avoid_put_inner_thoughts_in_kwargs else True
        return values

    @classmethod
    def default_config(cls, model_name: str):
        """
        Generate a default `LLMConfig` instance based on the provided model name.

        This class method takes a `model_name` as an argument and returns a predefined `LLMConfig` instance with
        appropriate settings for that specific model. It has configurations for well-known models like "gpt-4",
        "gpt-4o-mini", and "labo". If the provided `model_name` corresponds to an unsupported model, it raises
        a `ValueError`.

        Args:
        - `model_name`: A string representing the name of the model for which the default configuration is needed.

        Returns:
        - `LLMConfig`: A default `LLMConfig` instance with settings tailored to the specified model.

        Raises:
        - `ValueError`: If the provided `model_name` is for an unsupported model.
        """
        if model_name == "gpt-4":
            return cls(
                model="gpt-4",
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model_wrapper=None,
                context_window=8192,
            )
        elif model_name == "gpt-4o-mini":
            return cls(
                model="gpt-4o-mini",
                model_endpoint_type="openai",
                model_endpoint="https://api.openai.com/v1",
                model_wrapper=None,
                context_window=128000,
            )
        elif model_name == "labo":
            return cls(
                model="memgpt-openai",
                model_endpoint_type="openai",
                model_endpoint="https://inference.memgpt.ai",
                context_window=16384,
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def pretty_print(self) -> str:
        """
        Generate a human-readable string representation of the `LLMConfig` instance.

        This method constructs a string that includes the `model` name and, optionally, additional details like
        the `model_endpoint_type` and `model_endpoint` if they are set. This provides a convenient way to display
        the key aspects of the LLM configuration in a formatted manner.

        Returns:
        - `str`: A formatted string representing the `LLMConfig` instance.
        """
        return (
            f"{self.model}"
            + (f" [type={self.model_endpoint_type}]" if self.model_endpoint_type else "")
            + (f" [ip={self.model_endpoint}]" if self.model_endpoint else "")
        )