"""System prompts for RLM LangGraph."""

from typing import Any

RLM_SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant with access to a Python REPL environment. Your task is to answer questions by programmatically analyzing a large context that has been loaded into the environment.

## Context Information
- Context variable: `context` (a {context_type})
- Context length: {context_length:,} characters
{context_preview}

## Available Functions

### REPL Environment
You have access to a Python REPL with restricted but useful builtins. Write code in ```repl code blocks.

### LLM Query Functions
- `llm_query(prompt: str) -> str`: Query a sub-LLM for semantic analysis
- `llm_query_batched(prompts: list[str]) -> list[str]`: Batch multiple queries for efficiency

### Answer Functions
- `FINAL(answer: str)`: Call this when you have the final answer
- `FINAL_VAR(var_name: str)`: Use a variable's value as the final answer

### Pre-imported Modules
- `re`: Regular expressions
- `json`: JSON parsing
- `math`: Mathematical functions

You can also import: `datetime`, `collections`, `itertools`, `functools`, `operator`, `string`, `textwrap`, `difflib`, `statistics`, `random`, `copy`, `pprint`

## Strategy Guidelines

### For Long Contexts
1. **Explore first**: Check the context structure and size
   ```repl
   print(f"Context length: {{len(context)}} chars")
   print(f"First 500 chars: {{context[:500]}}")
   print(f"Last 500 chars: {{context[-500:]}}")
   ```

2. **Chunk and analyze**: Break into manageable pieces
   ```repl
   chunk_size = len(context) // 10
   for i in range(10):
       chunk = context[i*chunk_size:(i+1)*chunk_size]
       print(f"Chunk {{i}}: {{len(chunk)}} chars")
   ```

3. **Use sub-LLM for semantic analysis**: Let the sub-LLM analyze chunks
   ```repl
   results = []
   for i in range(10):
       chunk = context[i*chunk_size:(i+1)*chunk_size]
       result = llm_query(f"Summarize this text: {{chunk[:5000]}}")
       results.append(result)
   ```

4. **Use batching for efficiency**: Process multiple chunks in parallel
   ```repl
   prompts = [f"Find key facts in: {{context[i*size:(i+1)*size]}}" for i in range(10)]
   results = llm_query_batched(prompts)
   ```

### Search Strategies
- **Keyword search**: Use string methods or regex
  ```repl
  if "magic number" in context.lower():
      # Find and extract
      match = re.search(r"magic number is (\\d+)", context)
      if match:
          FINAL(match.group(1))
  ```

- **Binary search for needles**: Narrow down location
  ```repl
  def binary_search_contains(text, keyword):
      if keyword.lower() in text.lower():
          mid = len(text) // 2
          if len(text) < 1000:
              return text  # Found approximate location
          left = text[:mid+100]  # Overlap for safety
          right = text[mid-100:]
          if keyword.lower() in left.lower():
              return binary_search_contains(left, keyword)
          return binary_search_contains(right, keyword)
      return None
  ```

### Aggregation Strategies
- Collect findings from multiple chunks
- Use sub-LLM to synthesize results
- Build up understanding iteratively

## Important Notes
1. You can execute multiple code blocks across iterations
2. Variables persist between iterations
3. Always call FINAL() or FINAL_VAR() when you have the answer
4. If stuck, try a different approach (chunking, regex, sub-LLM analysis)
5. The context may be very large - don't try to print it all at once

## Your Task
Query: {query}

Begin by exploring the context structure, then develop a strategy to answer the query. Write your code in ```repl blocks.
"""


def build_system_prompt(
    query: str,
    context: str | dict | list,
    context_type: str,
    context_length: int,
    show_preview: bool = True,
) -> str:
    """
    Build the system prompt for the RLM.

    Args:
        query: The user's query
        context: The context data
        context_type: Type of context ("string", "json", "list")
        context_length: Length of context in characters
        show_preview: Whether to show a preview of the context

    Returns:
        Formatted system prompt
    """
    # Build context preview
    context_preview = ""
    if show_preview:
        preview_length = min(500, context_length)
        if isinstance(context, str):
            preview = context[:preview_length]
            if context_length > preview_length:
                preview += "..."
        elif isinstance(context, (dict, list)):
            import json

            preview = json.dumps(context, indent=2)[:preview_length]
            if len(json.dumps(context)) > preview_length:
                preview += "..."
        else:
            preview = str(context)[:preview_length]

        context_preview = f"- Context preview:\n```\n{preview}\n```"

    return RLM_SYSTEM_PROMPT_TEMPLATE.format(
        context_type=context_type,
        context_length=context_length,
        context_preview=context_preview,
        query=query,
    )


def build_initial_user_message(query: str) -> str:
    """
    Build the initial user message.

    Args:
        query: The user's query

    Returns:
        Formatted initial user message
    """
    return f"""Please answer this question by analyzing the context programmatically:

{query}

Start by exploring the context structure, then develop your analysis strategy. Write code in ```repl blocks."""


def build_iteration_message(
    iteration: int,
    stdout: str,
    stderr: str,
    error: str | None,
    locals_snapshot: dict[str, Any],
    sub_llm_responses: dict[str, str] | None = None,
) -> str:
    """
    Build a message for a new iteration with execution results.

    Args:
        iteration: Current iteration number
        stdout: Captured stdout from execution
        stderr: Captured stderr from execution
        error: Error message if any
        locals_snapshot: Snapshot of local variables
        sub_llm_responses: Responses from sub-LLM calls if any

    Returns:
        Formatted iteration message
    """
    parts = [f"[Iteration {iteration} Results]"]

    if stdout:
        parts.append(f"\n=== Output ===\n{stdout.rstrip()}")

    if stderr:
        parts.append(f"\n=== Stderr ===\n{stderr.rstrip()}")

    if error:
        parts.append(f"\n=== Error ===\n{error.rstrip()}")
        parts.append(
            "\nThe code raised an error. Please fix the issue and try again."
        )

    if sub_llm_responses:
        parts.append("\n=== Sub-LLM Responses ===")
        for call_id, response in sub_llm_responses.items():
            # Truncate long responses
            truncated = response[:1000] + "..." if len(response) > 1000 else response
            parts.append(f"{call_id}: {truncated}")

    # Show relevant variables
    if locals_snapshot:
        relevant_vars = {
            k: v
            for k, v in locals_snapshot.items()
            if k not in ("context", "re", "json", "math")
            and not k.startswith("_")
        }
        if relevant_vars:
            parts.append("\n=== Current Variables ===")
            for name, value in list(relevant_vars.items())[:10]:  # Limit to 10 vars
                value_str = repr(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                parts.append(f"{name} = {value_str}")

    if not error:
        parts.append(
            "\nContinue your analysis or call FINAL(answer) when you have the answer."
        )

    return "\n".join(parts)


def build_fallback_prompt(
    query: str,
    _execution_history: list[dict[str, Any]],
    accumulated_findings: list[str],
) -> str:
    """
    Build a prompt for generating a fallback answer after max iterations.

    Args:
        query: The original query
        execution_history: List of execution results
        accumulated_findings: List of findings from analysis

    Returns:
        Formatted fallback prompt
    """
    findings_text = "\n".join(f"- {f}" for f in accumulated_findings) if accumulated_findings else "No specific findings recorded."

    return f"""Maximum iterations reached. Based on your analysis so far, please provide your best answer to the original query.

Original Query: {query}

Findings from analysis:
{findings_text}

Please provide your best answer based on the information gathered. If you cannot answer definitively, explain what you discovered and any limitations."""
