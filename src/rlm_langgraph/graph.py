"""Main RLM LangGraph assembly."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    check_final_node,
    execute_code_node,
    extract_code_node,
    format_output_node,
    initialize_node,
)
from .nodes.llm_query import create_llm_query_node
from .nodes.output import create_fallback_node
from .nodes.sub_llm import create_sub_llm_handler_node
from .state import RLMState


def route_after_code_extraction(state: RLMState) -> Literal["execute_code", "check_final"]:
    """Route based on whether code blocks were extracted."""
    code_blocks = state.get("code_blocks", [])
    if code_blocks:
        return "execute_code"
    return "check_final"


def route_after_execution(state: RLMState) -> Literal["sub_llm_handler", "check_final"]:
    """Route based on whether there are pending sub-LLM calls."""
    pending_calls = state.get("pending_sub_llm_calls", [])
    if pending_calls:
        return "sub_llm_handler"
    return "check_final"


def route_after_final_check(state: RLMState) -> Literal["format_output", "fallback", "llm_query"]:
    """Route based on completion status and iteration count."""
    if state.get("is_complete", False):
        return "format_output"

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 30)

    if iteration >= max_iterations:
        return "fallback"

    return "llm_query"


_USE_DEFAULT_CHECKPOINTER = object()


def build_rlm_graph(
    root_llm: BaseChatModel,
    recursive_llm: BaseChatModel | None = None,
    checkpointer=_USE_DEFAULT_CHECKPOINTER,
):
    """
    Build the RLM LangGraph.

    Args:
        root_llm: The primary language model for reasoning
        recursive_llm: The model for sub-LLM calls (defaults to root_llm)
        checkpointer: LangGraph checkpointer for persistence.
            - Default: Uses MemorySaver
            - None: No checkpointer (for LangGraph API which handles persistence)
            - Custom checkpointer: Use provided checkpointer

    Returns:
        Compiled LangGraph ready for execution
    """
    if recursive_llm is None:
        recursive_llm = root_llm

    # Create node functions with bound LLMs
    llm_query_node = create_llm_query_node(root_llm)
    sub_llm_handler_node = create_sub_llm_handler_node(recursive_llm)
    fallback_node = create_fallback_node(root_llm)

    # Build the graph
    builder = StateGraph(RLMState)

    # Add nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("llm_query", llm_query_node)
    builder.add_node("extract_code", extract_code_node)
    builder.add_node("execute_code", execute_code_node)
    builder.add_node("sub_llm_handler", sub_llm_handler_node)
    builder.add_node("check_final", check_final_node)
    builder.add_node("format_output", format_output_node)
    builder.add_node("fallback", fallback_node)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "llm_query")
    builder.add_edge("llm_query", "extract_code")

    # Conditional: has code blocks?
    builder.add_conditional_edges(
        "extract_code",
        route_after_code_extraction,
        {
            "execute_code": "execute_code",
            "check_final": "check_final",
        },
    )

    # Conditional: has pending sub-LLM calls?
    builder.add_conditional_edges(
        "execute_code",
        route_after_execution,
        {
            "sub_llm_handler": "sub_llm_handler",
            "check_final": "check_final",
        },
    )

    builder.add_edge("sub_llm_handler", "check_final")

    # Conditional: is complete / max iterations?
    builder.add_conditional_edges(
        "check_final",
        route_after_final_check,
        {
            "format_output": "format_output",
            "fallback": "fallback",
            "llm_query": "llm_query",
        },
    )

    builder.add_edge("format_output", END)
    builder.add_edge("fallback", END)

    # Compile with optional checkpointer
    if checkpointer is _USE_DEFAULT_CHECKPOINTER:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def create_openai_rlm_graph(
    root_model: str = "gpt-4o",
    recursive_model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    checkpointer=_USE_DEFAULT_CHECKPOINTER,
):
    """
    Create an RLM graph using OpenAI models.

    Args:
        root_model: OpenAI model name for main reasoning
        recursive_model: OpenAI model name for sub-LLM calls
        temperature: LLM temperature
        checkpointer: Checkpointer for persistence (None = no checkpointer)

    Returns:
        Compiled LangGraph
    """
    from langchain_openai import ChatOpenAI

    root_llm = ChatOpenAI(model=root_model, temperature=temperature)
    recursive_llm = ChatOpenAI(model=recursive_model, temperature=temperature)

    return build_rlm_graph(
        root_llm=root_llm,
        recursive_llm=recursive_llm,
        checkpointer=checkpointer,
    )


def create_anthropic_rlm_graph(
    root_model: str = "claude-3-5-sonnet-20241022",
    recursive_model: str = "claude-3-5-haiku-20241022",
    temperature: float = 0.7,
    checkpointer=_USE_DEFAULT_CHECKPOINTER,
):
    """
    Create an RLM graph using Anthropic models.

    Args:
        root_model: Anthropic model name for main reasoning
        recursive_model: Anthropic model name for sub-LLM calls
        temperature: LLM temperature
        checkpointer: Checkpointer for persistence (None = no checkpointer)

    Returns:
        Compiled LangGraph
    """
    from langchain_anthropic import ChatAnthropic

    root_llm = ChatAnthropic(model=root_model, temperature=temperature)
    recursive_llm = ChatAnthropic(model=recursive_model, temperature=temperature)

    return build_rlm_graph(
        root_llm=root_llm,
        recursive_llm=recursive_llm,
        checkpointer=checkpointer,
    )
