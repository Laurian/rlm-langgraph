"""Test both OpenAI and Anthropic providers.

This script runs a simple query with both providers to verify they work correctly.
"""

import asyncio
import uuid

from dotenv import load_dotenv

from rlm_langgraph import (
    create_anthropic_rlm_graph,
    create_initial_state,
    create_openai_rlm_graph,
)

# Simple test context
CONTEXT = """
Product Catalog:

1. Widget Pro - $49.99
   - Color: Blue
   - Weight: 2.5 lbs
   - In stock: Yes

2. Gadget Plus - $79.99
   - Color: Silver
   - Weight: 1.2 lbs
   - In stock: No

3. Super Tool - $129.99
   - Color: Red
   - Weight: 3.8 lbs
   - In stock: Yes

4. Mini Device - $29.99
   - Color: Black
   - Weight: 0.5 lbs
   - In stock: Yes
"""


async def test_openai():
    """Test with OpenAI."""
    print("\n" + "=" * 60)
    print("Testing OpenAI (gpt-4o-mini)")
    print("=" * 60)

    graph = create_openai_rlm_graph(
        root_model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        temperature=0.0,
    )

    initial_state = create_initial_state(
        query="What is the total price of all in-stock products?",
        context=CONTEXT,
        max_iterations=8,
    )

    config = {"configurable": {"thread_id": f"test-openai-{uuid.uuid4()}"}}
    result = await graph.ainvoke(initial_state, config)

    print(f"Query: What is the total price of all in-stock products?")
    print(f"Answer: {result.get('final_answer', 'No answer')}")
    print(f"Iterations: {result.get('iteration', 0)}")
    print(f"Expected: $209.97 (Widget Pro + Super Tool + Mini Device)")

    return result


async def test_anthropic():
    """Test with Anthropic."""
    print("\n" + "=" * 60)
    print("Testing Anthropic (claude-3-5-haiku)")
    print("=" * 60)

    graph = create_anthropic_rlm_graph(
        root_model="claude-3-5-haiku-latest",
        recursive_model="claude-3-5-haiku-latest",
        temperature=0.0,
    )

    initial_state = create_initial_state(
        query="Which product is the heaviest and how much does it weigh?",
        context=CONTEXT,
        max_iterations=8,
    )

    config = {"configurable": {"thread_id": f"test-anthropic-{uuid.uuid4()}"}}
    result = await graph.ainvoke(initial_state, config)

    print(f"Query: Which product is the heaviest and how much does it weigh?")
    print(f"Answer: {result.get('final_answer', 'No answer')}")
    print(f"Iterations: {result.get('iteration', 0)}")
    print(f"Expected: Super Tool at 3.8 lbs")

    return result


async def main():
    load_dotenv()

    print("RLM Provider Test")
    print("Testing both OpenAI and Anthropic providers")

    # Test OpenAI
    openai_result = await test_openai()

    # Test Anthropic
    anthropic_result = await test_anthropic()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"OpenAI:    {'PASS' if openai_result.get('is_complete') else 'FAIL'}")
    print(f"Anthropic: {'PASS' if anthropic_result.get('is_complete') else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
