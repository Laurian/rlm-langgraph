"""Initialize node for RLM LangGraph."""

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import RLMState
from ..utils.prompts import build_initial_user_message, build_system_prompt


def initialize_node(state: RLMState) -> dict:
    """
    Initialize the RLM execution.

    This node:
    - Validates the context
    - Builds the system prompt with context metadata
    - Creates the initial message history

    Args:
        state: Current RLM state

    Returns:
        Updated state with messages initialized
    """
    query = state["query"]
    context = state["context"]
    context_type = state.get("context_type", "string")
    context_length = state.get("context_length", 0)

    # Build system prompt
    system_prompt = build_system_prompt(
        query=query,
        context=context,
        context_type=context_type,
        context_length=context_length,
        show_preview=True,
    )

    # Build initial user message
    user_message = build_initial_user_message(query)

    # Initialize messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    return {
        "messages": messages,
        "iteration": 0,
        "is_complete": False,
    }
