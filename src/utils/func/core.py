from typing import Optional

from labo.agent import Agent
from labo.constants import RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE
from labo.utils import json_dumps


def send_message(self: "Agent", message: str) -> Optional[str]:
    """
    Send a message through the agent's interface. Currently, it simply forwards the message
    via the `assistant_message` method of the interface and returns `None`.

    :param self: The Agent instance.
    :param message: The message to be sent.
    :return: `None` as there's no return value from the message sending operation for now.
    """
    self.interface.assistant_message(message)
    return None


def conversation_search(self: "Agent", query: str, page: Optional[int] = 0) -> Optional[str]:
    """
    Perform a conversation search based on the provided query for the given agent.
    It fetches a page of user messages related to the query and formats the results.

    :param self: The Agent instance.
    :param query: The search query string.
    :param page: The page number to retrieve (default is 0). Must be an integer, otherwise a ValueError is raised.
    :return: A string representing the search results, including information about the number of results and the page.
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except ValueError:
        raise ValueError(f"'page' argument must be an integer")

    messages = self.message_manager.list_user_messages_for_agent(
        agent_id=self.agent_state.id,
        actor=self.user,
        query_text=query,
        limit=RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
    )
    total = len(messages)
    num_pages = max(0, (total // RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE))
    if len(messages) == 0:
        results_str = f"No results found."
    else:
        results_pref = f"Showing {len(messages)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [message.text for message in messages]
        results_str = f"{results_pref} {json_dumps(results_formatted)}"
    return results_str


def archival_memory_insert(self: "Agent", content: str) -> Optional[str]:
    """
    Insert content into the archival memory of the agent.

    :param self: The Agent instance.
    :param content: The content to be inserted.
    :return: `None` after successfully inserting the content into the archival memory.
    """
    self.passage_manager.insert_passage(
        agent_state=self.agent_state,
        agent_id=self.agent_state.id,
        text=content,
        actor=self.user,
    )
    return None


def archival_memory_search(self: "Agent", query: str, page: Optional[int] = 0, start: Optional[int] = 0) -> Optional[str]:
    """
    Perform a search in the archival memory of the agent based on the provided query.
    It retrieves a page of results starting from the specified position and formats them.

    :param self: The Agent instance.
    :param query: The search query string.
    :param page: The page number to retrieve (default is 0). Must be an integer, otherwise a ValueError is raised.
    :param start: The starting index for the results (default is 0).
    :return: A list of formatted results (dictionaries containing timestamp and content) or raises an exception if an error occurs during retrieval.
    """
    if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
        page = 0
    try:
        page = int(page)
    except ValueError:
        raise ValueError(f"'page' argument must be an integer")

    try:
        all_results = self.agent_manager.list_passages(
            actor=self.user,
            agent_id=self.agent_state.id,
            query_text=query,
            limit=RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE + start,
            embedding_config=self.agent_state.embedding_config,
            embed_query=True,
        )

        end = min(RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE + start, len(all_results))
        paged_results = all_results[start:end]

        formatted_results = [{"timestamp": str(result.created_at), "content": result.text} for result in paged_results]

        return formatted_results
    except Exception as e:
        raise e


def core_memory_append(agent_state: "AgentState", label: str, content: str) -> Optional[str]:
    """
    Append content to the specified core memory block of the agent state.

    :param agent_state: The AgentState instance representing the state of the agent.
    :param label: The label of the core memory block to which content will be appended.
    :param content: The content to be appended.
    :return: `None` after successfully appending the content to the core memory block.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    new_value = current_value + "\n" + str(content)
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None


def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> Optional[str]:
    """
    Replace old content with new content in the specified core memory block of the agent state.
    Raises a ValueError if the old content is not found in the memory block.

    :param agent_state: The AgentState instance representing the state of the agent.
    :param label: The label of the core memory block where the replacement will happen.
    :param old_content: The content to be replaced.
    :param new_content: The new content to replace the old content.
    :return: `None` after successfully replacing the content in the core memory block.
    """
    current_value = str(agent_state.memory.get_block(label).value)
    if old_content not in current_value:
        raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    new_value = current_value.replace(str(old_content), str(new_content))
    agent_state.memory.update_block_value(label=label, value=new_value)
    return None