import requests
from labo.utils import printd, smart_urljoin


def mistral_get_model_list(url: str, api_key: str) -> dict:
    """
    Send a GET request to the specified URL (constructed by joining the base URL with "models") to retrieve the list of Mistral models.

    This function sets the appropriate headers (including the authorization header if an API key is provided) and makes the request.
    It handles different types of exceptions that might occur during the request process, logging relevant information and re-raising the exceptions.

    Args:
        url (str): The base URL to which the "models" endpoint will be joined.
        api_key (str): The API key for authorization (can be None if not required).

    Returns:
        dict: The JSON response containing the list of models if the request is successful.

    Raises:
        requests.exceptions.HTTPError: If an HTTP error occurs during the request (e.g., 4xx or 5xx status codes).
        requests.exceptions.RequestException: If any other request-related exception occurs (e.g., connection issues).
        Exception: If an unknown exception occurs during the process.
    """
    # Construct the full URL for the models endpoint
    full_url = smart_urljoin(url, "models")

    # Set the headers, including the authorization header if API key is provided
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    printd(f"Sending request to {full_url}")
    try:
        # Make the GET request
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        printd(f"Got HTTPError, exception={http_err}, response={response.text if response else None}")
        raise
    except requests.exceptions.RequestException as req_err:
        printd(f"Got RequestException, exception={req_err}, response={response.text if response else None}")
        raise
    except Exception as e:
        printd(f"Got unknown Exception, exception={e}, response={response.text if response else None}")
        raise