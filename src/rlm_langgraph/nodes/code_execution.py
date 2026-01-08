"""Code extraction and execution nodes for RLM LangGraph."""

from langchain_core.messages import HumanMessage

from ..repl import REPLEnvironment
from ..state import ExecutionResult, RLMState, SubLLMCall
from ..utils.parsing import extract_code_blocks
from ..utils.prompts import build_iteration_message


def extract_code_node(state: RLMState) -> dict:
    """
    Extract code blocks from the LLM response.

    This node:
    - Parses the last LLM response for ```repl code blocks
    - Stores extracted code blocks in state

    Args:
        state: Current RLM state

    Returns:
        Updated state with extracted code blocks
    """
    last_response = state.get("last_response", "")

    # Extract code blocks
    code_blocks = extract_code_blocks(last_response, language="repl")

    # Also try python blocks as fallback
    if not code_blocks:
        code_blocks = extract_code_blocks(last_response, language="python")

    # Extract just the code strings
    code_strings = [block.code for block in code_blocks]

    return {
        "code_blocks": code_strings,
    }


def execute_code_node(state: RLMState) -> dict:
    """
    Execute extracted code blocks in the REPL environment.

    This node:
    - Creates or retrieves the REPL environment
    - Executes each code block
    - Captures stdout, stderr, errors
    - Collects any sub-LLM calls made during execution
    - Updates message history with results

    Args:
        state: Current RLM state

    Returns:
        Updated state with execution results
    """
    code_blocks = state.get("code_blocks", [])
    context = state["context"]
    repl_locals = state.get("repl_locals", {})

    if not code_blocks:
        return {}

    # Create REPL environment with context
    repl = REPLEnvironment(context)

    # Restore any previous locals (excluding special items)
    for key, value in repl_locals.items():
        if key not in ("__builtins__", "context", "llm_query", "llm_query_batched", "FINAL", "FINAL_VAR"):
            repl.locals[key] = value

    execution_results: list[ExecutionResult] = []
    all_pending_calls: list[SubLLMCall] = []
    final_answer = None
    final_var_name = None

    # Execute each code block
    for code in code_blocks:
        result = repl.execute(code)

        # Convert to ExecutionResult format
        exec_result = ExecutionResult(
            code=result.code,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.error,
            sub_llm_calls=[
                SubLLMCall(
                    call_id=call.call_id,
                    prompt=call.prompt,
                    model=call.model,
                )
                for call in result.sub_llm_calls
            ],
            locals_snapshot=result.locals_snapshot,
        )
        execution_results.append(exec_result)

        # Collect sub-LLM calls
        for call in result.sub_llm_calls:
            all_pending_calls.append(
                SubLLMCall(
                    call_id=call.call_id,
                    prompt=call.prompt,
                    model=call.model,
                )
            )

        # Check for final answer
        if result.final_answer is not None:
            final_answer = result.final_answer
        if result.final_var_name is not None:
            final_var_name = result.final_var_name

    # Get the final answer value if FINAL_VAR was used
    if final_var_name and final_answer is None and final_var_name in repl.locals:
        final_answer = str(repl.locals[final_var_name])

    # Build iteration message for message history
    combined_stdout = "\n".join(r["stdout"] for r in execution_results if r["stdout"])
    combined_stderr = "\n".join(r["stderr"] for r in execution_results if r["stderr"])
    combined_errors = "\n".join(r["error"] for r in execution_results if r["error"])

    iteration_message = build_iteration_message(
        iteration=state.get("iteration", 0),
        stdout=combined_stdout,
        stderr=combined_stderr,
        error=combined_errors if combined_errors else None,
        locals_snapshot=repl.locals,
        sub_llm_responses=None,  # Will be filled in by sub_llm_handler
    )

    # Save REPL locals for persistence
    # Filter out non-serializable items
    serializable_locals = {}
    for key, value in repl.locals.items():
        if key.startswith("_") or key == "__builtins__":
            continue
        if callable(value) and key in ("llm_query", "llm_query_batched", "FINAL", "FINAL_VAR"):
            continue
        try:
            import json
            json.dumps(value)
            serializable_locals[key] = value
        except (TypeError, ValueError):
            # Store repr for non-serializable
            serializable_locals[key] = f"<{type(value).__name__}>"

    result_update = {
        "execution_results": execution_results,
        "pending_sub_llm_calls": all_pending_calls,
        "repl_locals": serializable_locals,
        "messages": [HumanMessage(content=iteration_message)],
    }

    if final_answer is not None:
        result_update["final_answer"] = final_answer
        result_update["is_complete"] = True

    return result_update
