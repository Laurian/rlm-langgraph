"""LLM Query node for RLM LangGraph."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from ..state import RLMState


async def llm_query_node(
    state: RLMState,
    llm: BaseChatModel,
) -> dict:
    """
    Query the LLM for the next action.

    This node:
    - Invokes the LLM with the current message history
    - Increments the iteration counter
    - Adds the response to message history

    Args:
        state: Current RLM state
        llm: The language model to use

    Returns:
        Updated state with LLM response
    """
    messages = state["messages"]
    iteration = state.get("iteration", 0)

    # Query the LLM
    response = await llm.ainvoke(messages)

    # Extract content from response
    response_content = response.content if isinstance(response, AIMessage) else str(response)

    return {
        "messages": [AIMessage(content=response_content)],
        "last_response": response_content,
        "iteration": iteration + 1,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }


def create_llm_query_node(llm: BaseChatModel):
    """
    Create an LLM query node with the specified model.

    Args:
        llm: The language model to use

    Returns:
        Async function that can be used as a LangGraph node
    """

    async def node(state: RLMState) -> dict:
        return await llm_query_node(state, llm)

    return node
