"""Output formatting nodes for RLM LangGraph."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ..state import RLMState
from ..utils.prompts import build_fallback_prompt


def format_output_node(state: RLMState) -> dict:
    """
    Format the final output.

    This node:
    - Packages the final answer
    - Includes execution statistics

    Args:
        state: Current RLM state

    Returns:
        Updated state with formatted output
    """
    final_answer = state.get("final_answer", "No answer found.")

    return {
        "final_answer": final_answer,
        "is_complete": True,
    }


async def fallback_node(
    state: RLMState,
    llm: BaseChatModel,
) -> dict:
    """
    Generate a fallback answer when max iterations reached.

    This node:
    - Summarizes the analysis performed so far
    - Asks the LLM to provide its best answer based on findings
    - Returns a fallback answer

    Args:
        state: Current RLM state
        llm: The language model to use

    Returns:
        Updated state with fallback answer
    """
    query = state["query"]
    execution_results = state.get("execution_results", [])

    # Collect any findings from stdout
    accumulated_findings = []
    for result in execution_results:
        stdout = result.get("stdout", "")
        if stdout:
            # Extract meaningful lines (skip empty and too long)
            for line in stdout.split("\n"):
                line = line.strip()
                if line and len(line) < 500:
                    accumulated_findings.append(line)

    # Build fallback prompt
    fallback_prompt = build_fallback_prompt(
        query=query,
        execution_history=execution_results,
        accumulated_findings=accumulated_findings[:20],  # Limit findings
    )

    # Query LLM for fallback answer
    response = await llm.ainvoke([HumanMessage(content=fallback_prompt)])

    fallback_answer = response.content if hasattr(response, 'content') else str(response)

    return {
        "final_answer": f"[Fallback - max iterations reached]\n{fallback_answer}",
        "is_complete": True,
        "messages": [
            HumanMessage(content=fallback_prompt),
            AIMessage(content=fallback_answer),
        ],
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }


def create_fallback_node(llm: BaseChatModel):
    """
    Create a fallback node with the specified model.

    Args:
        llm: The language model to use for fallback generation

    Returns:
        Async function that can be used as a LangGraph node
    """

    async def node(state: RLMState) -> dict:
        return await fallback_node(state, llm)

    return node
