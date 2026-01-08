"""RLM State Definition for LangGraph."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SubLLMCall(TypedDict):
    """A pending sub-LLM call to be processed."""

    call_id: str
    prompt: str
    model: str | None  # None means use default recursive_model


class ExecutionResult(TypedDict):
    """Result from executing a code block in the REPL."""

    code: str
    stdout: str
    stderr: str
    error: str | None
    sub_llm_calls: list[SubLLMCall]
    locals_snapshot: dict[str, Any]  # Snapshot of relevant variables


class RLMState(TypedDict, total=False):
    """
    State for the Recursive Language Model LangGraph.

    This state tracks the entire RLM execution including:
    - Input query and context
    - Message history for multi-turn reasoning
    - REPL environment state
    - Iteration and depth tracking
    - Code execution results
    - Sub-LLM call handling
    - Final answer
    """

    # ===== Input =====
    query: str  # The user's query/question
    context: str | dict | list  # The context to reason over
    context_type: str  # "string", "json", or "list"
    context_length: int  # Total character count

    # ===== Message History =====
    # Using add_messages reducer for proper message handling
    messages: Annotated[list, add_messages]

    # ===== REPL Environment State =====
    repl_locals: dict[str, Any]  # Variables in REPL namespace

    # ===== Iteration Tracking =====
    iteration: int  # Current iteration (0-indexed)
    max_iterations: int  # Maximum iterations before fallback

    # ===== Depth Tracking (for recursion) =====
    depth: int  # Current recursion depth (0 = root)
    max_depth: int  # Maximum recursion depth

    # ===== Code Execution =====
    last_response: str  # Last LLM response
    code_blocks: list[str]  # Extracted code blocks from response
    execution_results: list[ExecutionResult]  # Results from code execution

    # ===== Sub-LLM Call Handling =====
    pending_sub_llm_calls: list[SubLLMCall]  # Calls to process
    sub_llm_responses: dict[str, str]  # call_id -> response

    # ===== Final Answer =====
    final_answer: str | None  # The final answer if found
    is_complete: bool  # Whether RLM has completed

    # ===== Configuration =====
    root_model: str  # Model name for root LLM
    recursive_model: str  # Model name for sub-LLM calls
    temperature: float  # LLM temperature

    # ===== Metrics =====
    total_llm_calls: int  # Total LLM API calls made
    total_tokens_used: int  # Approximate token usage


def create_initial_state(
    query: str,
    context: str | dict | list,
    root_model: str = "gpt-4o",
    recursive_model: str = "gpt-4o-mini",
    max_iterations: int = 30,
    max_depth: int = 1,
    temperature: float = 0.7,
) -> RLMState:
    """
    Create an initial RLM state with the given inputs.

    Args:
        query: The user's query/question
        context: The context to reason over (string, dict, or list)
        root_model: Model name for the root LLM
        recursive_model: Model name for sub-LLM calls
        max_iterations: Maximum iterations before fallback
        max_depth: Maximum recursion depth
        temperature: LLM temperature

    Returns:
        Initialized RLMState ready for graph execution
    """
    # Determine context type and length
    if isinstance(context, str):
        context_type = "string"
        context_length = len(context)
    elif isinstance(context, dict):
        context_type = "json"
        import json

        context_length = len(json.dumps(context))
    elif isinstance(context, list):
        context_type = "list"
        context_length = sum(len(str(item)) for item in context)
    else:
        context_type = "string"
        context = str(context)
        context_length = len(context)

    return RLMState(
        # Input
        query=query,
        context=context,
        context_type=context_type,
        context_length=context_length,
        # Message history (empty initially, system message added in initialize node)
        messages=[],
        # REPL state
        repl_locals={},
        # Iteration tracking
        iteration=0,
        max_iterations=max_iterations,
        # Depth tracking
        depth=0,
        max_depth=max_depth,
        # Code execution
        last_response="",
        code_blocks=[],
        execution_results=[],
        # Sub-LLM handling
        pending_sub_llm_calls=[],
        sub_llm_responses={},
        # Final answer
        final_answer=None,
        is_complete=False,
        # Configuration
        root_model=root_model,
        recursive_model=recursive_model,
        temperature=temperature,
        # Metrics
        total_llm_calls=0,
        total_tokens_used=0,
    )
