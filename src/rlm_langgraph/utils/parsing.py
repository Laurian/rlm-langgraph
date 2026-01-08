"""Parsing utilities for RLM code extraction and result formatting."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CodeBlock:
    """A code block extracted from LLM response."""

    code: str
    language: str
    start_pos: int
    end_pos: int


def extract_code_blocks(response: str, language: str = "repl") -> list[CodeBlock]:
    """
    Extract code blocks from an LLM response.

    Looks for markdown-style code blocks with the specified language tag.

    Args:
        response: The LLM response text
        language: The language tag to look for (default: "repl")

    Returns:
        List of CodeBlock objects containing the extracted code
    """
    blocks = []

    # Pattern to match ```language ... ``` blocks
    # Supports both ```repl and ```python (which we'll treat as repl)
    pattern = rf"```(?:{language}|python)\s*\n(.*?)```"

    for match in re.finditer(pattern, response, re.DOTALL):
        code = match.group(1).strip()
        if code:  # Only add non-empty blocks
            blocks.append(
                CodeBlock(
                    code=code,
                    language=language,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

    return blocks


def find_final_in_code(code: str) -> tuple[str | None, str | None]:
    """
    Find FINAL() or FINAL_VAR() calls in code.

    Args:
        code: Python code to analyze

    Returns:
        Tuple of (final_answer, final_var_name) where one or both may be None
    """
    final_answer = None
    final_var_name = None

    # Look for FINAL("answer") or FINAL('answer') or FINAL(variable)
    final_pattern = r'FINAL\s*\(\s*(["\'])(.*?)\1\s*\)'
    final_match = re.search(final_pattern, code)
    if final_match:
        final_answer = final_match.group(2)

    # Look for FINAL(variable) without quotes
    final_var_pattern = r"FINAL\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)"
    final_var_match = re.search(final_var_pattern, code)
    if final_var_match and not final_match:
        # This is FINAL(variable_name), need to evaluate at runtime
        final_var_name = final_var_match.group(1)

    # Look for FINAL_VAR("var_name") or FINAL_VAR('var_name')
    final_var_str_pattern = r'FINAL_VAR\s*\(\s*(["\'])([a-zA-Z_][a-zA-Z0-9_]*)\1\s*\)'
    final_var_str_match = re.search(final_var_str_pattern, code)
    if final_var_str_match:
        final_var_name = final_var_str_match.group(2)

    return final_answer, final_var_name


def format_execution_result(
    code: str,
    stdout: str,
    stderr: str,
    error: str | None,
    locals_snapshot: dict[str, Any],
) -> str:
    """
    Format code execution results for inclusion in message history.

    Args:
        code: The executed code
        stdout: Captured standard output
        stderr: Captured standard error
        error: Error message if execution failed
        locals_snapshot: Snapshot of relevant local variables

    Returns:
        Formatted string representation of the execution result
    """
    parts = []

    # Code section
    parts.append("=== Code Executed ===")
    parts.append(f"```python\n{code}\n```")

    # Output section
    if stdout:
        parts.append("\n=== Output ===")
        parts.append(stdout.rstrip())

    # Stderr section (if any)
    if stderr:
        parts.append("\n=== Stderr ===")
        parts.append(stderr.rstrip())

    # Error section (if any)
    if error:
        parts.append("\n=== Error ===")
        parts.append(error.rstrip())

    # Variables section (show non-trivial variables)
    if locals_snapshot:
        relevant_vars = {
            k: v
            for k, v in locals_snapshot.items()
            if k not in ("context", "re", "json", "math")
            and not k.startswith("_")
            and not callable(v)
        }
        if relevant_vars:
            parts.append("\n=== Variables ===")
            for name, value in relevant_vars.items():
                value_str = repr(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                parts.append(f"{name} = {value_str}")

    return "\n".join(parts)


def format_iteration_prompt(
    iteration: int,
    execution_results: list[dict[str, Any]],
    has_pending_calls: bool = False,
) -> str:
    """
    Format the iteration prompt for continued reasoning.

    Args:
        iteration: Current iteration number
        execution_results: List of execution result dictionaries
        has_pending_calls: Whether there are pending sub-LLM calls

    Returns:
        Formatted prompt for the next iteration
    """
    parts = [f"[Iteration {iteration}]"]

    if execution_results:
        for i, result in enumerate(execution_results):
            if len(execution_results) > 1:
                parts.append(f"\n--- Block {i + 1} ---")
            parts.append(
                format_execution_result(
                    code=result.get("code", ""),
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    error=result.get("error"),
                    locals_snapshot=result.get("locals_snapshot", {}),
                )
            )

    if has_pending_calls:
        parts.append(
            "\n[Note: Sub-LLM calls are being processed. "
            "Results will be available in the next iteration.]"
        )

    parts.append(
        "\nContinue your analysis. "
        "Write more code to explore the context, or call FINAL(answer) when ready."
    )

    return "\n".join(parts)


def truncate_context_preview(context: str | dict | list, max_length: int = 500) -> str:
    """
    Create a truncated preview of the context for display.

    Args:
        context: The context to preview
        max_length: Maximum length of the preview

    Returns:
        Truncated string representation
    """
    if isinstance(context, str):
        if len(context) <= max_length:
            return context
        return context[:max_length] + f"... [{len(context) - max_length} more chars]"
    elif isinstance(context, (dict, list)):
        import json

        context_str = json.dumps(context, indent=2)
        if len(context_str) <= max_length:
            return context_str
        return (
            context_str[:max_length]
            + f"... [{len(context_str) - max_length} more chars]"
        )
    else:
        return str(context)[:max_length]
