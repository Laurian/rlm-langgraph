"""Final answer check node for RLM LangGraph."""

import re

from ..state import RLMState


def check_final_node(state: RLMState) -> dict:
    """
    Check if a final answer has been provided.

    This node:
    - Checks if final_answer was set during code execution
    - Also scans the last response for FINAL() patterns
    - Sets is_complete flag if answer is found

    Args:
        state: Current RLM state

    Returns:
        Updated state with final answer status
    """
    # Check if already marked complete
    if state.get("is_complete", False):
        return {}

    final_answer = state.get("final_answer")
    if final_answer is not None:
        return {
            "is_complete": True,
        }

    # Check last response for any FINAL indicators we might have missed
    last_response = state.get("last_response", "")

    # Look for explicit "The answer is X" or "FINAL: X" patterns
    # that might appear outside of code blocks
    patterns = [
        r"FINAL\s*:\s*(.+?)(?:\n|$)",
        r"The\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\n|$)",
        r"Answer[:\s]+(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, last_response, re.IGNORECASE)
        if match:
            potential_answer = match.group(1).strip()
            # Only accept if it looks like a real answer (not a template)
            if potential_answer and not potential_answer.startswith(("```", "{")):
                return {
                    "final_answer": potential_answer,
                    "is_complete": True,
                }

    return {}
