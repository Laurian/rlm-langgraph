# RLM-LangGraph

A LangGraph implementation of **Recursive Language Models (RLMs)** - a general inference strategy that enables LLMs to process arbitrarily long prompts by treating them as part of an external environment.

## Paper Reference

This implementation is based on the paper:

> **Recursive Language Models**
>
> Alex L. Zhang, Tim Kraska, Omar Khattab
>
> MIT CSAIL
>
> arXiv:2512.24601v1 31 Dec 2025
>
> *We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt.*

## Key Innovation

Instead of feeding long prompts directly into the neural network, RLMs:

1. **Load context as a variable** in a Python REPL environment
2. **Allow the LLM to write code** to peek into, decompose, and analyze the context
3. **Enable recursive sub-LLM calls** for semantic analysis of context chunks
4. **Iterate until a final answer** is found via `FINAL()` or `FINAL_VAR()`

This enables:
- Handling inputs **up to 100x beyond model context windows**
- **Dramatically outperforming** base LLMs on long-context tasks
- **Comparable or lower cost** per query

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install from source:

```bash
git clone https://github.com/yourusername/rlm-langgraph.git
cd rlm-langgraph
uv sync
```

This creates a virtual environment and installs all dependencies. Run examples with:

```bash
uv run python examples/basic_usage.py
```

### Using pip

```bash
git clone https://github.com/yourusername/rlm-langgraph.git
cd rlm-langgraph
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from rlm_langgraph import build_rlm_graph

async def main():
    # Build the RLM graph
    graph = build_rlm_graph(
        root_model="gpt-4o",           # Main reasoning model
        recursive_model="gpt-4o-mini", # Model for sub-LLM calls
        max_iterations=30,
    )

    # Your long context (e.g., a large document)
    context = open("large_document.txt").read()

    # Run the RLM
    result = await graph.ainvoke({
        "query": "What are the main themes discussed in this document?",
        "context": context,
    })

    print(result["final_answer"])

asyncio.run(main())
```

### Needle in a Haystack Example

```python
import asyncio
import random
from rlm_langgraph import build_rlm_graph

async def needle_in_haystack():
    # Generate a large haystack with a hidden needle
    magic_number = random.randint(1000000, 9999999)

    haystack_lines = []
    needle_position = random.randint(0, 99999)

    for i in range(100000):
        if i == needle_position:
            haystack_lines.append(f"The magic number is {magic_number}.")
        else:
            haystack_lines.append(f"Line {i}: This is irrelevant filler text.")

    context = "\n".join(haystack_lines)
    print(f"Context size: {len(context):,} characters")
    print(f"Expected answer: {magic_number}")

    # Build and run the RLM
    graph = build_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
    )

    result = await graph.ainvoke({
        "query": "Find the magic number hidden in the text.",
        "context": context,
    })

    print(f"RLM Answer: {result['final_answer']}")
    print(f"Iterations: {result['iteration']}")

asyncio.run(needle_in_haystack())
```

### Using Different LLM Providers

```python
from rlm_langgraph import build_rlm_graph

# OpenAI
graph = build_rlm_graph(
    root_model="gpt-4o",
    recursive_model="gpt-4o-mini",
)

# Anthropic
graph = build_rlm_graph(
    root_model="claude-3-5-sonnet-20241022",
    recursive_model="claude-3-5-haiku-20241022",
)
```

### With Persistence (Checkpointing)

```python
from langgraph.checkpoint.memory import MemorySaver
from rlm_langgraph import build_rlm_graph

# Use memory saver for checkpoints
checkpointer = MemorySaver()

graph = build_rlm_graph(
    root_model="gpt-4o",
    recursive_model="gpt-4o-mini",
    checkpointer=checkpointer,
)

# Run with a thread ID for resumability
config = {"configurable": {"thread_id": "my-session-123"}}
result = await graph.ainvoke(initial_state, config)

# Later: inspect or resume from checkpoint
state = graph.get_state(config)
```

## LangGraph Studio (Interactive Visualization)

You can run the RLM graph interactively using LangGraph Studio, which provides a visual interface for exploring graph execution, inspecting state, and debugging.

### Setup

1. Install dev dependencies:

```bash
uv sync --all-extras
```

2. Add your LangSmith API key to `.env` (recommended for tracing):

```bash
LANGSMITH_API_KEY=lsv2_...
```

3. Start the dev server:

```bash
uv run langgraph dev
```

4. Open http://localhost:8123 in your browser

### Example Input

Use this sample input to test the graph:

```json
{
  "query": "What is the total price of all in-stock products?",
  "context": "Product Catalog:\n1. Widget Pro - $49.99 (In stock: Yes)\n2. Gadget Plus - $79.99 (In stock: No)\n3. Super Tool - $129.99 (In stock: Yes)"
}
```

This example demonstrates:
- Code execution in the REPL (parsing prices, filtering in-stock items)
- The iterative reasoning loop
- Final answer via `FINAL()` call

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RLM LangGraph                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  START → Initialize → LLM Query ←──────────────────┐    │
│                          │                         │    │
│                          ▼                         │    │
│                    Extract Code                    │    │
│                          │                         │    │
│              ┌───────────┴───────────┐             │    │
│              ▼                       ▼             │    │
│         Execute Code           Check Final        │    │
│              │                       │            │    │
│              ▼                       │            │    │
│        Sub-LLM Handler               │            │    │
│              │                       │            │    │
│              └───────────┬───────────┘            │    │
│                          │                        │    │
│              ┌───────────┴───────────┐            │    │
│              ▼                       ▼            │    │
│        Has Final?              Max Iterations?    │    │
│              │                       │            │    │
│         Yes  │                  No   │   Yes      │    │
│              ▼                       │            │    │
│        Format Output ◄───────────────┘            │    │
│              │                                    │    │
│              ▼                                    │    │
│             END                                   │    │
│                                                   │    │
└───────────────────────────────────────────────────┘
```

### The REPL Environment

The LLM has access to a Python REPL with:

- **`context`** - The input context as a variable
- **`llm_query(prompt)`** - Query a sub-LLM for semantic analysis
- **`llm_query_batched(prompts)`** - Batch multiple sub-LLM queries
- **`FINAL(answer)`** - Return the final answer
- **`FINAL_VAR(var_name)`** - Return a variable as the final answer

### Example LLM Strategy

The LLM might generate code like this:

```python
# 1. Examine the context
print(f"Context length: {len(context)} characters")
print(f"First 500 chars: {context[:500]}")

# 2. Chunk and analyze
chunk_size = len(context) // 10
findings = []

for i in range(10):
    chunk = context[i*chunk_size:(i+1)*chunk_size]
    result = llm_query(f"Find any magic numbers in: {chunk}")
    findings.append(result)
    print(f"Chunk {i}: {result}")

# 3. Aggregate and answer
final = llm_query(f"Based on these findings, what is the magic number? {findings}")
FINAL(final)
```

## Configuration

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Recommended for langgraph dev (enables tracing in LangSmith Studio)
LANGSMITH_API_KEY=lsv2_...
```

### Graph Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_model` | `"gpt-4o"` | Model for main reasoning |
| `recursive_model` | `"gpt-4o-mini"` | Model for sub-LLM calls |
| `max_iterations` | `30` | Maximum reasoning iterations |
| `max_depth` | `1` | Recursion depth (1 = simple sub-calls) |
| `temperature` | `0.7` | LLM temperature |
| `checkpointer` | `MemorySaver()` | LangGraph checkpointer |

## Benchmarks

Based on the original paper, RLMs achieve:

| Task | Base GPT-5 | RLM(GPT-5) |
|------|------------|------------|
| S-NIAH (1M tokens) | ~60% | **98%** |
| OOLONG | 44% | **56.5%** |
| OOLONG-Pairs | 0.04% | **58%** |
| BrowseComp+ (1K) | 0%* | **91.3%** |

*Base model exceeds context window

## License

MIT

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025}
}
```

## Acknowledgments

- Original paper authors: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)
- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
