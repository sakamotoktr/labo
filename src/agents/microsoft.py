from collections import defaultdict
import requests
from labo.agents.helpers import make_post_request
from labo.schemas.llm_config import LLMConfig
from labo.schemas.openai.chat_completion_response import ChatCompletionResponse
from labo.schemas.openai.chat_completions import ChatCompletionRequest
from labo.schemas.openai.embedding_response import EmbeddingResponse
from labo.settings import ModelSettings


class AzureModelContextLengthMapper:
    """
    A class to manage the mapping between Azure models and their corresponding context lengths.
    This class provides methods to access, modify, and query the mapping.
    """

    def __init__(self):
        """
        Initialize the mapping with the predefined set of Azure models and their context lengths.
        """
        self.model_to_context_length = {
            "babbage-002": 16384,
            "davinci-002": 16384,
            "gpt-35-turbo-0613": 4096,
            "gpt-35-turbo-1106": 16385,
            "gpt-35-turbo-0125": 16385,
            "gpt-4-0613": 8192,
            "gpt-4o-mini-2024-07-18": 128000,
            "gpt-4o-2024-08-06": 128000,
        }

    def get_context_length(self, model_name):
        """
        Get the context length for a given Azure model.

        Args:
            model_name (str): The name of the Azure model.

        Returns:
            int: The context length of the model if it exists in the mapping, otherwise raises a KeyError.
        """
        return self.model_to_context_length[model_name]

    def add_model(self, model_name, context_length):
        """
        Add a new Azure model and its corresponding context length to the mapping.

        Args:
            model_name (str): The name of the new Azure model.
            context_length (int): The context length of the new model.
        """
        self.model_to_context_length[model_name] = context_length

    def has_model(self, model_name):
        """
        Check if a given Azure model exists in the mapping.

        Args:
            model_name (str): The name of the Azure model.

        Returns:
            bool: True if the model exists in the mapping, False otherwise.
        """
        return model_name in self.model_to_context_length

    def get_all_models(self):
        """
        Get a list of all the Azure models in the mapping.

        Returns:
            list: A list containing the names of all the Azure models in the mapping.
        """
        return list(self.model_to_context_length.keys())

    def update_model_context_length(self, model_name, new_context_length):
        """
        Update the context length of an existing Azure model in the mapping.

        Args:
            model_name (str): The name of the existing Azure model.
            new_context_length (int): The new context length to set for the model.

        Raises:
            KeyError: If the model does not exist in the mapping.
        """
        if model_name not in self.model_to_context_length:
            raise KeyError(f"Model {model_name} not found in the mapping.")
        self.model_to_context_length[model_name] = new_context_length


# Example usage:
mapper = AzureModelContextLengthMapper()
try:
    print(mapper.get_context_length("gpt-35-turbo-1106"))
    mapper.add_model("new-model", 10000)
    print(mapper.has_model("new-model"))
    print(mapper.get_all_models())
    mapper.update_model_context_length("gpt-35-turbo-1106", 18000)
    print(mapper.get_context_length("gpt-35-turbo-1106"))
except KeyError as e:
    print(f"Error: {e}")


def get_azure_chat_completions_endpoint(base_url: str, model: str, api_version: str):
    """
    Generate the endpoint URL for Azure OpenAI chat completions.

    This function constructs the URL for making chat completions requests to a specific
    model deployed on Azure OpenAI, using the provided base URL, model name, and API version.

    :param base_url: The base URL of the Azure OpenAI service.
    :param model: The name of the deployed model.
    :param api_version: The version of the API to use.
    :return: The complete URL for the chat completions endpoint.
    """
    return f"{base_url}/openai/deployments/{model}/chat/completions?api-version={api_version}"


def get_azure_embeddings_endpoint(base_url: str, model: str, api_version: str):
    """
    Generate the endpoint URL for Azure OpenAI embeddings.

    This function constructs the URL for retrieving embeddings from a specific
    model deployed on Azure OpenAI, using the provided base URL, model name, and API version.

    :param base_url: The base URL of the Azure OpenAI service.
    :param model: The name of the deployed model.
    :param api_version: The version of the API to use.
    :return: The complete URL for the embeddings endpoint.
    """
    return f"{base_url}/openai/deployments/{model}/embeddings?api-version={api_version}"


def get_azure_model_list_endpoint(base_url: str, api_version: str):
    """
    Generate the endpoint URL for retrieving the list of available Azure OpenAI models.

    This function constructs the URL for fetching information about all the models
    available on the Azure OpenAI service, using the provided base URL and API version.

    :param base_url: The base URL of the Azure OpenAI service.
    :param api_version: The version of the API to use.
    :return: The complete URL for the model list endpoint.
    """
    return f"{base_url}/openai/models?api-version={api_version}"


def get_azure_deployment_list_endpoint(base_url: str):
    """
    Generate the endpoint URL for retrieving the list of deployments on Azure OpenAI.

    This function constructs the URL for getting details about all the deployments
    on the Azure OpenAI service, using a fixed API version as per the requirement.

    :param base_url: The base URL of the Azure OpenAI service.
    :return: The complete URL for the deployment list endpoint.
    """
    return f"{base_url}/openai/deployments?api-version=2023-03-15-preview"


def azure_openai_get_deployed_model_list(
    base_url: str, api_key: str, api_version: str
) -> list:
    """
    Retrieve the list of deployed models on Azure OpenAI.

    This function first fetches the list of all available models and then the list of
    deployed models. It then filters the available models to only include those that
    are actually deployed. Finally, it keeps only the latest version of each deployed
    model based on the creation date.

    :param base_url: The base URL of the Azure OpenAI service.
    :param api_key: The API key for authentication.
    :param api_version: The version of the API to use.
    :return: A list of the latest deployed models.
    :raises RuntimeError: If there is an error while retrieving the model list or deployment list.
    """
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["api-key"] = f"{api_key}"

    # Fetch the list of all available models
    url = get_azure_model_list_endpoint(base_url, api_version)
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to retrieve model list: {e}")
    all_available_models = response.json().get("data", [])

    # Fetch the list of deployed models
    url = get_azure_deployment_list_endpoint(base_url)
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to retrieve model list: {e}")

    deployed_models = response.json().get("data", [])
    deployed_model_names = set([m["id"] for m in deployed_models])

    # Filter available models to only include deployed ones
    deployed_models = [
        m for m in all_available_models if m["id"] in deployed_model_names
    ]

    # Keep only the latest version of each deployed model
    latest_models = defaultdict()
    for model in deployed_models:
        model_id = model["id"]
        updated_at = model["created_at"]

        if (
            model_id not in latest_models
            or updated_at > latest_models[model_id]["created_at"]
        ):
            latest_models[model_id] = model

    return list(latest_models.values())


def azure_openai_get_chat_completion_model_list(
    base_url: str, api_key: str, api_version: str
) -> list:
    """
    Retrieve the list of deployed models on Azure OpenAI that support chat completions.

    This function first gets the list of all deployed models and then filters it to
    include only those models that have the 'chat_completion' capability enabled.

    :param base_url: The base URL of the Azure OpenAI service.
    :param api_key: The API key for authentication.
    :param api_version: The version of the API to use.
    :return: A list of deployed models that support chat completions.
    """
    model_list = azure_openai_get_deployed_model_list(base_url, api_key, api_version)
    model_options = [
        m for m in model_list if m.get("capabilities").get("chat_completion") == True
    ]
    return model_options


def azure_openai_get_embeddings_model_list(
    base_url: str,
    api_key: str,
    api_version: str,
    require_embedding_in_name: bool = True,
) -> list:
    """
    Retrieve the list of deployed models on Azure OpenAI that support embeddings.

    This function first gets the list of all deployed models and then filters it based
    on whether the model has the 'embeddings' capability enabled and optionally, if the
    model name contains the word 'embedding'.

    :param base_url: The base URL of the Azure OpenAI service.
    :param api_key: The API key for authentication.
    :param api_version: The version of the API to use.
    :param require_embedding_in_name: Whether to require the model name to contain 'embedding' (default is True).
    :return: A list of deployed models that support embeddings.
    """

    def valid_embedding_model(m: dict):
        """
        Check if a given model is a valid embedding model based on certain criteria.

        This inner function checks if the model has the 'embeddings' capability enabled
        and optionally, if the model name contains the word 'embedding'.

        :param m: A dictionary representing a model's information.
        :return: True if the model is a valid embedding model, False otherwise.
        """
        valid_name = True
        if require_embedding_in_name:
            valid_name = "embedding" in m["id"]

        return m.get("capabilities").get("embeddings") == True and valid_name

    model_list = azure_openai_get_deployed_model_list(base_url, api_key, api_version)
    model_options = [m for m in model_list if valid_embedding_model(m)]
    return model_options


def azure_openai_chat_completions_request(
    model_settings: ModelSettings,
    llm_config: LLMConfig,
    api_key: str,
    chat_completion_request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Make a chat completions request to Azure OpenAI.

    This function sends a chat completions request to the specified Azure OpenAI model,
    after preparing the request data and headers. It also handles some optional fields
    in the request data and validates that the API key is provided.

    :param model_settings: An object containing settings related to the Azure model (e.g., base URL, API version).
    :param llm_config: An object containing the configuration for the language model (e.g., model name).
    :param api_key: The API key for authentication.
    :param chat_completion_request: An object containing the details of the chat completions request.
    :return: A ChatCompletionResponse object representing the response from the Azure OpenAI service.
    :raises AssertionError: If the API key is not provided.
    """
    assert api_key is not None, "Missing required field when calling Azure OpenAI"

    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}
    data = chat_completion_request.model_dump(exclude_none=True)

    # Remove 'functions' and related fields if they are None
    if "functions" in data and data["functions"] is None:
        data.pop("functions")
        data.pop("function_call", None)

    # Remove 'tools' and related fields if they are None
    if "tools" in data and data["tools"] is None:
        data.pop("tools")
        data.pop("tool_choice", None)

    url = get_azure_chat_completions_endpoint(
        model_settings.azure_base_url,
        llm_config.model,
        model_settings.azure_api_version,
    )
    response_json = make_post_request(url, headers, data)

    # Set content to None if it's missing in the response
    if "content" not in response_json["choices"][0].get("message"):
        response_json["choices"][0]["message"]["content"] = None
    response = ChatCompletionResponse(**response_json)
    return response


def azure_openai_embeddings_request(
    resource_name: str, deployment_id: str, api_version: str, api_key: str, data: dict
) -> EmbeddingResponse:
    """
    Make an embeddings request to Azure OpenAI.

    This function sends a request to the specified Azure OpenAI deployment to retrieve
    embeddings, using the provided data, headers, and URL constructed with the given
    resource name, deployment ID, and API version.

    :param resource_name: The name of the Azure resource.
    :param deployment_id: The ID of the deployment to use for the embeddings request.
    :param api_version: The version of the API to use.
    :param api_key: The API key for authentication.
    :param data: A dictionary containing the data for the embeddings request.
    :return: An EmbeddingResponse object representing the response from the Azure OpenAI service.
    """
    url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/embeddings?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": f"{api_key}"}

    response_json = make_post_request(url, headers, data)
    return EmbeddingResponse(**response_json)
