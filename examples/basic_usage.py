"""Basic usage example for RLM LangGraph.

This example demonstrates how to use the RLM to answer questions
about a document by programmatically analyzing it.
"""

import asyncio
import uuid

from dotenv import load_dotenv

from rlm_langgraph import create_initial_state, create_openai_rlm_graph


async def main():
    # Load environment variables (OPENAI_API_KEY)
    load_dotenv()

    # Sample context - a short article
    context = """
    The History of Artificial Intelligence

    Artificial Intelligence (AI) has a rich history dating back to the mid-20th century.
    The term "Artificial Intelligence" was first coined by John McCarthy in 1956 at the
    Dartmouth Conference, which is widely considered the birth of AI as a field.

    Early AI research focused on symbolic reasoning and problem-solving. Programs like
    the Logic Theorist (1956) and the General Problem Solver (1959) demonstrated that
    machines could perform tasks that required human-like reasoning.

    The 1960s and 1970s saw the development of expert systems, which encoded human
    knowledge into rules. MYCIN, developed at Stanford in 1972, could diagnose bacterial
    infections and recommend antibiotics with accuracy comparable to human experts.

    The field experienced two major "AI winters" - periods of reduced funding and interest.
    The first occurred in the 1970s after initial promises failed to materialize, and
    the second in the late 1980s after the expert systems bubble burst.

    The modern era of AI began with the resurgence of neural networks. In 2012, AlexNet
    demonstrated the power of deep learning by winning the ImageNet competition with
    unprecedented accuracy. This sparked the current AI boom.

    Key milestones in recent years include:
    - 2016: AlphaGo defeats world champion Go player Lee Sedol
    - 2017: Transformers architecture introduced, revolutionizing NLP
    - 2020: GPT-3 demonstrates unprecedented language capabilities
    - 2022: ChatGPT brings conversational AI to mainstream audiences
    - 2023: GPT-4 achieves human-level performance on many benchmarks

    Today, AI is integrated into countless applications, from virtual assistants to
    autonomous vehicles, medical diagnosis to scientific research. The field continues
    to advance rapidly, with new breakthroughs occurring regularly.

    The future of AI remains a topic of intense debate. While optimists see AI as a
    tool that will solve humanity's greatest challenges, others worry about potential
    risks including job displacement, privacy concerns, and the long-term implications
    of artificial general intelligence (AGI).

    Total words in this document: approximately 350
    Author: AI History Research Team
    Last updated: 2024
    """

    # Build the RLM graph
    graph = create_openai_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
        temperature=0.7,
    )

    # Create initial state
    initial_state = create_initial_state(
        query="What year was the term 'Artificial Intelligence' first coined, and who coined it?",
        context=context,
        max_iterations=10,
    )

    # Run the graph
    config = {"configurable": {"thread_id": f"rlm-{uuid.uuid4()}"}}

    print("Running RLM...")
    print(f"Query: {initial_state['query']}")
    print(f"Context length: {initial_state['context_length']} characters")
    print("-" * 50)

    result = await graph.ainvoke(initial_state, config)

    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print(result.get("final_answer", "No answer found"))
    print("=" * 50)
    print(f"\nIterations: {result.get('iteration', 0)}")
    print(f"Total LLM calls: {result.get('total_llm_calls', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
