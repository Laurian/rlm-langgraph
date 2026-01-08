"""Needle in a Haystack example for RLM LangGraph.

This example demonstrates the RLM's ability to find specific information
hidden within a large amount of irrelevant text - a classic benchmark
from the original paper.
"""

import asyncio
import random
import uuid

from dotenv import load_dotenv

from rlm_langgraph import create_initial_state, create_openai_rlm_graph


def generate_haystack(num_lines: int = 10000, needle_position: int | None = None) -> tuple[str, int, int]:
    """
    Generate a haystack with a hidden needle.

    Args:
        num_lines: Number of lines in the haystack
        needle_position: Position to place the needle (random if None)

    Returns:
        Tuple of (haystack_text, magic_number, needle_position)
    """
    magic_number = random.randint(1000000, 9999999)

    if needle_position is None:
        needle_position = random.randint(0, num_lines - 1)

    lines = []
    for i in range(num_lines):
        if i == needle_position:
            lines.append(f"The magic number is {magic_number}.")
        else:
            # Generate various types of filler content
            filler_types = [
                f"Line {i}: This is irrelevant filler text that serves no purpose.",
                f"Entry {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                f"Record {i}: Data point with random value {random.randint(1, 1000)}.",
                f"Note {i}: Nothing important to see here, move along.",
                f"Item {i}: Standard placeholder content for testing purposes.",
            ]
            lines.append(random.choice(filler_types))

    return "\n".join(lines), magic_number, needle_position


async def main():
    # Load environment variables
    load_dotenv()

    # Generate a haystack with configurable size
    print("Generating haystack...")
    num_lines = 10000  # Adjust this to make the task harder/easier
    haystack, magic_number, needle_position = generate_haystack(num_lines)

    print("Haystack generated:")
    print(f"  - Total lines: {num_lines:,}")
    print(f"  - Total characters: {len(haystack):,}")
    print(f"  - Needle position: line {needle_position:,}")
    print(f"  - Expected answer: {magic_number}")
    print("-" * 50)

    # Build the RLM graph
    graph = create_openai_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
        temperature=0.7,
    )

    # Create initial state
    initial_state = create_initial_state(
        query="Find the magic number hidden in the text. Return only the number.",
        context=haystack,
        max_iterations=20,
    )

    # Run the graph
    config = {"configurable": {"thread_id": f"rlm-needle-{uuid.uuid4()}"}}

    print("\nRunning RLM to find the needle...")
    print(f"Query: {initial_state['query']}")
    print("-" * 50)

    result = await graph.ainvoke(initial_state, config)

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"  Expected answer: {magic_number}")
    print(f"  RLM answer: {result.get('final_answer', 'No answer found')}")

    # Check if correct
    final_answer = result.get("final_answer", "")
    if str(magic_number) in str(final_answer):
        print("  Status: CORRECT!")
    else:
        print("  Status: INCORRECT")

    print("=" * 50)
    print("\nStats:")
    print(f"  Iterations: {result.get('iteration', 0)}")
    print(f"  Total LLM calls: {result.get('total_llm_calls', 0)}")


async def benchmark(num_trials: int = 5, haystack_size: int = 10000):
    """
    Run multiple trials to measure accuracy.

    Args:
        num_trials: Number of trials to run
        haystack_size: Number of lines in each haystack
    """
    load_dotenv()

    print(f"Running {num_trials} trials with haystack size {haystack_size:,}...")
    print("=" * 50)

    graph = create_openai_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
        temperature=0.7,
    )

    correct = 0
    total_iterations = 0
    total_llm_calls = 0

    for trial in range(num_trials):
        haystack, magic_number, needle_position = generate_haystack(haystack_size)

        initial_state = create_initial_state(
            query="Find the magic number hidden in the text. Return only the number.",
            context=haystack,
            max_iterations=20,
        )

        config = {"configurable": {"thread_id": f"rlm-bench-{uuid.uuid4()}"}}

        result = await graph.ainvoke(initial_state, config)

        final_answer = result.get("final_answer", "")
        is_correct = str(magic_number) in str(final_answer)

        if is_correct:
            correct += 1

        total_iterations += result.get("iteration", 0)
        total_llm_calls += result.get("total_llm_calls", 0)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"Trial {trial + 1}: {status} (expected {magic_number}, got {final_answer})")

    print("=" * 50)
    print(f"Accuracy: {correct}/{num_trials} ({100 * correct / num_trials:.1f}%)")
    print(f"Avg iterations: {total_iterations / num_trials:.1f}")
    print(f"Avg LLM calls: {total_llm_calls / num_trials:.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        asyncio.run(benchmark(num_trials=num_trials))
    else:
        asyncio.run(main())
