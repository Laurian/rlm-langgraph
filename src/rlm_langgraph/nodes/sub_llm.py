"""Sub-LLM handler node for RLM LangGraph."""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..state import RLMState


async def _execute_single_call(
    llm: BaseChatModel,
    call_id: str,
    prompt: str,
) -> tuple[str, str]:
    """
    Execute a single sub-LLM call.

    Args:
        llm: The language model to use
        call_id: Unique identifier for this call
        prompt: The prompt to send to the LLM

    Returns:
        Tuple of (call_id, response)
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return call_id, response.content
    except Exception as e:
        return call_id, f"[Error: {str(e)}]"


async def sub_llm_handler_node(
    state: RLMState,
    recursive_llm: BaseChatModel,
) -> dict:
    """
    Process pending sub-LLM calls in parallel.

    This node:
    - Takes all pending sub-LLM calls from the state
    - Executes them in parallel using asyncio.gather
    - Stores responses back into state
    - Injects responses into the REPL namespace

    Args:
        state: Current RLM state
        recursive_llm: The language model for sub-LLM calls

    Returns:
        Updated state with sub-LLM responses
    """
    pending_calls = state.get("pending_sub_llm_calls", [])

    if not pending_calls:
        return {}

    # Execute all calls in parallel
    tasks = [
        _execute_single_call(
            llm=recursive_llm,
            call_id=call["call_id"],
            prompt=call["prompt"],
        )
        for call in pending_calls
    ]

    results = await asyncio.gather(*tasks)

    # Build responses dictionary
    sub_llm_responses = dict(results)

    # Update the REPL locals by replacing placeholders
    repl_locals = state.get("repl_locals", {}).copy()

    for key, value in list(repl_locals.items()):
        if isinstance(value, str):
            for call_id, response in sub_llm_responses.items():
                placeholder = f"<<PENDING_LLM_CALL:{call_id}>>"
                if placeholder in value:
                    repl_locals[key] = value.replace(placeholder, response)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, str):
                    new_item = item
                    for call_id, response in sub_llm_responses.items():
                        placeholder = f"<<PENDING_LLM_CALL:{call_id}>>"
                        if placeholder in new_item:
                            new_item = new_item.replace(placeholder, response)
                    new_list.append(new_item)
                else:
                    new_list.append(item)
            repl_locals[key] = new_list

    # Build message about sub-LLM responses
    response_summary_parts = ["[Sub-LLM Responses Received]"]
    for call_id, response in sub_llm_responses.items():
        truncated = response[:500] + "..." if len(response) > 500 else response
        response_summary_parts.append(f"\n{call_id}:\n{truncated}")

    response_summary_parts.append(
        "\nThe sub-LLM responses have been injected into your variables. "
        "Continue your analysis or call FINAL(answer) when you have the answer."
    )

    return {
        "sub_llm_responses": sub_llm_responses,
        "pending_sub_llm_calls": [],  # Clear pending calls
        "repl_locals": repl_locals,
        "messages": [HumanMessage(content="\n".join(response_summary_parts))],
        "total_llm_calls": state.get("total_llm_calls", 0) + len(pending_calls),
    }


def create_sub_llm_handler_node(recursive_llm: BaseChatModel):
    """
    Create a sub-LLM handler node with the specified model.

    Args:
        recursive_llm: The language model for sub-LLM calls

    Returns:
        Async function that can be used as a LangGraph node
    """

    async def node(state: RLMState) -> dict:
        return await sub_llm_handler_node(state, recursive_llm)

    return node
