"""RLM LangGraph - Recursive Language Models as a LangGraph implementation.

Based on the paper "Recursive Language Models" (arXiv:2512.24601v1)
by Alex L. Zhang, Tim Kraska, Omar Khattab from MIT CSAIL.
"""

from .graph import (
    build_rlm_graph,
    create_anthropic_rlm_graph,
    create_openai_rlm_graph,
)
from .repl import SAFE_BUILTINS, REPLEnvironment, REPLResult
from .state import ExecutionResult, RLMState, SubLLMCall, create_initial_state

__version__ = "0.1.0"

__all__ = [
    # State
    "RLMState",
    "create_initial_state",
    "SubLLMCall",
    "ExecutionResult",
    # Graph builders
    "build_rlm_graph",
    "create_openai_rlm_graph",
    "create_anthropic_rlm_graph",
    # REPL
    "REPLEnvironment",
    "REPLResult",
    "SAFE_BUILTINS",
]
