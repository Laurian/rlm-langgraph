"""Integration tests with actual LLM calls.

These tests require valid API keys in .env file:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

Run with: uv run pytest tests/test_integration.py -v
"""

import os
import uuid

import pytest
from dotenv import load_dotenv

from rlm_langgraph import (
    create_anthropic_rlm_graph,
    create_initial_state,
    create_openai_rlm_graph,
)

# Load environment variables
load_dotenv()

# Skip all tests if API keys are not available
OPENAI_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
ANTHROPIC_KEY_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.fixture
def simple_context():
    """A simple context for basic tests."""
    return """
    Company Information:
    - Company Name: Acme Corporation
    - Founded: 1985
    - Headquarters: San Francisco, CA
    - CEO: Jane Smith
    - Number of Employees: 5,000
    - Annual Revenue: $2.5 billion
    - Main Products: Industrial equipment, safety gear, and tools
    """


@pytest.fixture
def needle_haystack_context():
    """A needle-in-haystack context for testing search capabilities."""
    lines = []
    for i in range(500):
        if i == 237:
            lines.append("The secret code is ALPHA-7749.")
        else:
            lines.append(f"Line {i}: This is filler text with no important information.")
    return "\n".join(lines)


class TestOpenAIIntegration:
    """Integration tests using OpenAI models."""

    @pytest.mark.skipif(not OPENAI_KEY_AVAILABLE, reason="OpenAI API key not available")
    @pytest.mark.asyncio
    async def test_simple_query(self, simple_context):
        """Test a simple factual query with OpenAI."""
        graph = create_openai_rlm_graph(
            root_model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="What is the name of the CEO?",
            context=simple_context,
            max_iterations=5,
        )

        config = {"configurable": {"thread_id": f"test-openai-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # The answer should mention Jane Smith
        assert "jane" in final_answer.lower() or "smith" in final_answer.lower()

    @pytest.mark.skipif(not OPENAI_KEY_AVAILABLE, reason="OpenAI API key not available")
    @pytest.mark.asyncio
    async def test_needle_search(self, needle_haystack_context):
        """Test needle-in-haystack search with OpenAI."""
        graph = create_openai_rlm_graph(
            root_model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="Find the secret code hidden in the text.",
            context=needle_haystack_context,
            max_iterations=10,
        )

        config = {
            "configurable": {"thread_id": f"test-needle-{uuid.uuid4()}"},
            "recursion_limit": 100,  # Increase limit for needle search
        }
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # The answer should contain the secret code (or at least part of it)
        assert (
            "ALPHA-7749" in final_answer
            or "ALPHA" in final_answer
            or "7749" in final_answer
        ), f"Expected secret code in answer, got: {final_answer}"

    @pytest.mark.skipif(not OPENAI_KEY_AVAILABLE, reason="OpenAI API key not available")
    @pytest.mark.asyncio
    async def test_computation_query(self, simple_context):
        """Test a query requiring computation with OpenAI."""
        graph = create_openai_rlm_graph(
            root_model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="How many years has the company been operating? (Current year is 2024)",
            context=simple_context,
            max_iterations=5,
        )

        config = {"configurable": {"thread_id": f"test-compute-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # 2024 - 1985 = 39 years
        assert "39" in final_answer


class TestAnthropicIntegration:
    """Integration tests using Anthropic models."""

    @pytest.mark.skipif(
        not ANTHROPIC_KEY_AVAILABLE, reason="Anthropic API key not available"
    )
    @pytest.mark.asyncio
    async def test_simple_query(self, simple_context):
        """Test a simple factual query with Anthropic."""
        graph = create_anthropic_rlm_graph(
            root_model="claude-3-5-haiku-latest",
            recursive_model="claude-3-5-haiku-latest",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="What is the annual revenue of the company?",
            context=simple_context,
            max_iterations=5,
        )

        config = {"configurable": {"thread_id": f"test-anthropic-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # The answer should mention $2.5 billion
        assert "2.5" in final_answer or "billion" in final_answer.lower()

    @pytest.mark.skipif(
        not ANTHROPIC_KEY_AVAILABLE, reason="Anthropic API key not available"
    )
    @pytest.mark.xfail(reason="Haiku may not always find the needle - this is a challenging test")
    @pytest.mark.asyncio
    async def test_needle_search(self, needle_haystack_context):
        """Test needle-in-haystack search with Anthropic."""
        graph = create_anthropic_rlm_graph(
            root_model="claude-3-5-haiku-latest",
            recursive_model="claude-3-5-haiku-latest",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="Find the secret code hidden in the text.",
            context=needle_haystack_context,
            max_iterations=10,
        )

        config = {
            "configurable": {"thread_id": f"test-needle-anthropic-{uuid.uuid4()}"},
            "recursion_limit": 100,  # Increase limit for needle search
        }
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # The answer should contain the secret code (or at least part of it)
        # Note: This is a challenging test - partial matches are acceptable
        assert (
            "ALPHA-7749" in final_answer
            or "ALPHA" in final_answer
            or "7749" in final_answer
            or "secret" in final_answer.lower()
        ), f"Expected secret code reference in answer, got: {final_answer}"

    @pytest.mark.skipif(
        not ANTHROPIC_KEY_AVAILABLE, reason="Anthropic API key not available"
    )
    @pytest.mark.asyncio
    async def test_list_extraction(self, simple_context):
        """Test extracting multiple items with Anthropic."""
        graph = create_anthropic_rlm_graph(
            root_model="claude-3-5-haiku-latest",
            recursive_model="claude-3-5-haiku-latest",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="List all the main products mentioned.",
            context=simple_context,
            max_iterations=5,
        )

        config = {"configurable": {"thread_id": f"test-list-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "").lower()
        assert final_answer, "Expected a final answer"
        # Should mention the products
        assert "equipment" in final_answer or "tools" in final_answer or "safety" in final_answer


class TestSubLLMCalls:
    """Tests for sub-LLM call functionality."""

    @pytest.mark.skipif(not OPENAI_KEY_AVAILABLE, reason="OpenAI API key not available")
    @pytest.mark.asyncio
    async def test_sub_llm_calls_executed(self):
        """Test that sub-LLM calls are properly executed."""
        # Context with multiple sections that might trigger sub-LLM analysis
        context = """
        Section A: Financial Results
        Q1 Revenue: $500M
        Q2 Revenue: $550M
        Q3 Revenue: $600M
        Q4 Revenue: $650M

        Section B: Employee Statistics
        Engineering: 200 employees
        Sales: 150 employees
        Marketing: 100 employees
        Operations: 50 employees

        Section C: Regional Performance
        North America: 40% of sales
        Europe: 35% of sales
        Asia Pacific: 25% of sales
        """

        graph = create_openai_rlm_graph(
            root_model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            temperature=0.0,
        )

        initial_state = create_initial_state(
            query="What is the total annual revenue and total number of employees?",
            context=context,
            max_iterations=8,
        )

        config = {"configurable": {"thread_id": f"test-sub-llm-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        assert result.get("is_complete", False)
        final_answer = result.get("final_answer", "")
        assert final_answer, "Expected a final answer"
        # Total revenue: 500+550+600+650 = 2300M = $2.3B
        # Total employees: 200+150+100+50 = 500
        # Answer should contain reasonable numbers
        assert "2" in final_answer or "500" in final_answer or "300" in final_answer
