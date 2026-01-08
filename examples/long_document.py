"""Long document processing example for RLM LangGraph.

This example demonstrates how the RLM handles analysis of
longer documents by chunking and using sub-LLM calls.
"""

import asyncio
import uuid

from dotenv import load_dotenv

from rlm_langgraph import create_initial_state, create_openai_rlm_graph

# Sample long document (a fictional research paper abstract collection)
LONG_DOCUMENT = """
# Collection of Research Paper Abstracts on Machine Learning

## Paper 1: Deep Learning for Natural Language Processing
Authors: Smith, J., Johnson, M., Williams, K.
Published: 2023

Abstract: This paper presents a comprehensive survey of deep learning techniques
applied to natural language processing tasks. We review transformer architectures,
attention mechanisms, and pre-training strategies. Our analysis covers 150 papers
published between 2018 and 2023. We find that transformer-based models have
largely replaced recurrent neural networks for most NLP tasks. Key contributions
include BERT, GPT, and T5 architectures. The paper concludes with recommendations
for practitioners and identifies open research questions including model efficiency,
multilingual capabilities, and reasoning abilities.

Keywords: deep learning, NLP, transformers, attention, BERT, GPT

---

## Paper 2: Reinforcement Learning in Robotics
Authors: Chen, L., Park, S., Anderson, R.
Published: 2023

Abstract: We present a novel approach to robot manipulation using deep reinforcement
learning with sparse rewards. Our method combines curiosity-driven exploration with
hierarchical skill learning. Experiments on the OpenAI Gym environments show a 45%
improvement in sample efficiency compared to baseline methods. We demonstrate
successful transfer to real-world robotic arms for pick-and-place tasks. The key
innovation is a learned world model that enables planning without extensive
real-world interaction. Limitations include high computational costs and sensitivity
to hyperparameter choices.

Keywords: reinforcement learning, robotics, manipulation, sparse rewards, world models

---

## Paper 3: Federated Learning for Healthcare Applications
Authors: Martinez, A., Brown, T., Lee, H.
Published: 2024

Abstract: Privacy-preserving machine learning is critical for healthcare applications.
This paper introduces FedHealth, a federated learning framework designed for
heterogeneous medical data. We address challenges including non-IID data distribution,
communication efficiency, and differential privacy guarantees. Evaluation on chest
X-ray classification across 12 hospitals shows comparable accuracy to centralized
training while maintaining patient privacy. The framework reduces communication
overhead by 60% through gradient compression. Clinical deployment considerations
and regulatory compliance are discussed.

Keywords: federated learning, healthcare, privacy, differential privacy, medical imaging

---

## Paper 4: Graph Neural Networks for Drug Discovery
Authors: Wilson, E., Kumar, P., Zhang, Y.
Published: 2023

Abstract: Molecular property prediction is essential for accelerating drug discovery.
We propose MolGraph, a graph neural network architecture optimized for molecular
representations. The model achieves state-of-the-art results on 8 of 12 MoleculeNet
benchmarks. Key innovations include edge-aware message passing and attention-based
readout functions. We demonstrate practical utility through a case study identifying
promising candidates for a novel cancer treatment. The code and pre-trained models
are publicly available. Future work will explore multi-task learning and uncertainty
quantification.

Keywords: graph neural networks, drug discovery, molecular properties, chemistry

---

## Paper 5: Self-Supervised Learning for Computer Vision
Authors: Thompson, D., Garcia, M., Patel, R.
Published: 2024

Abstract: Self-supervised pre-training has transformed computer vision. We present
VisualSSL, a unified framework for comparing contrastive and non-contrastive methods.
Our experiments span ImageNet, COCO, and specialized domains including satellite
imagery and medical imaging. Key findings: (1) contrastive methods excel with
limited labeled data, (2) non-contrastive methods show better transfer to out-of-
distribution tasks, (3) combining both approaches yields additional gains. We
provide practical guidelines for practitioners choosing pre-training strategies.
The framework includes 20 implemented methods with standardized evaluation protocols.

Keywords: self-supervised learning, computer vision, contrastive learning, pre-training

---

## Paper 6: Large Language Models for Code Generation
Authors: Davis, C., Miller, S., Taylor, J.
Published: 2024

Abstract: Code generation models have reached remarkable capabilities. This paper
evaluates 15 large language models on programming tasks across 12 languages. We
introduce CodeBench, a comprehensive benchmark covering algorithm implementation,
debugging, code translation, and documentation generation. Results show GPT-4
leading on most tasks, with open-source models closing the gap rapidly. We analyze
failure modes including logic errors, syntax mistakes, and security vulnerabilities.
Notably, models struggle with tasks requiring long-range dependencies or domain-
specific knowledge. Recommendations for safe deployment in development environments
are provided.

Keywords: code generation, large language models, programming, benchmarks, software engineering

---

## Paper 7: Efficient Transformers: A Survey
Authors: Robinson, A., White, L., Harris, B.
Published: 2023

Abstract: The quadratic complexity of standard transformers limits their application
to long sequences. This survey categorizes efficiency improvements into four classes:
sparse attention, low-rank approximations, recurrence, and memory mechanisms. We
provide unified comparisons across 35 efficient transformer variants on language
modeling and long-range arena benchmarks. Key recommendations: (1) Performer excels
for very long sequences, (2) Longformer balances efficiency and accuracy for documents,
(3) Flash Attention provides drop-in improvements for standard transformers. The
survey includes complexity analysis and practical implementation considerations.

Keywords: transformers, efficiency, attention, long sequences, complexity

---

## Paper 8: Multimodal Learning: Vision and Language
Authors: Adams, N., Clark, E., Scott, P.
Published: 2024

Abstract: Integrating visual and linguistic understanding remains challenging. We
present VLMM (Vision-Language Multimodal Model), a unified architecture for image
captioning, visual question answering, and image-text retrieval. The model uses
a shared encoder-decoder with modality-specific adaptors. Training combines
contrastive pre-training on 400M image-text pairs with supervised fine-tuning.
VLMM achieves state-of-the-art on VQA v2 (82.4%), COCO Captions (148.3 CIDEr),
and Flickr30k retrieval. Analysis reveals emergent compositional reasoning abilities.
Limitations include poor handling of fine-grained spatial relationships and
counting tasks.

Keywords: multimodal learning, vision-language, image captioning, VQA, retrieval

---

## Paper 9: Neural Architecture Search: Methods and Applications
Authors: Young, H., King, W., Hall, D.
Published: 2023

Abstract: Automating neural architecture design has matured significantly. This
paper reviews neural architecture search (NAS) methods including evolutionary
algorithms, reinforcement learning, and gradient-based approaches. We categorize
search spaces, discuss weight-sharing strategies, and analyze computational costs.
Practical applications in image classification, object detection, and NLP are
covered. Key insight: once-for-all networks provide good accuracy-efficiency
trade-offs with minimal search cost. We identify remaining challenges including
transferability across tasks and search space design. Guidelines for practitioners
considering NAS adoption are provided.

Keywords: neural architecture search, AutoML, evolutionary algorithms, efficiency

---

## Paper 10: Trustworthy AI: Fairness, Robustness, and Interpretability
Authors: Moore, V., Jackson, T., Wright, A.
Published: 2024

Abstract: Deploying AI systems responsibly requires addressing fairness, robustness,
and interpretability. This paper presents a unified framework for evaluating and
improving trustworthiness. We propose TrustScore, a composite metric balancing
multiple dimensions. Experiments on credit scoring, hiring, and medical diagnosis
demonstrate trade-offs between fairness constraints and accuracy. For robustness,
we evaluate adversarial training, certified defenses, and out-of-distribution
detection. Interpretability methods are compared including SHAP, LIME, and concept
bottlenecks. Case studies illustrate practical deployment considerations. The
paper concludes with policy recommendations and future research directions.

Keywords: trustworthy AI, fairness, robustness, interpretability, responsible AI

---

## Summary Statistics
- Total papers: 10
- Publication years: 2023 (5), 2024 (5)
- Primary topics: NLP (2), Computer Vision (2), Healthcare (1), Drug Discovery (1),
  Code Generation (1), Efficiency (1), Multimodal (1), AutoML (1), Trustworthy AI (1)
- Average keywords per paper: 5.2
- Most common author institution: Not specified

## Citation Statistics (as of 2024)
- Paper 1 (Deep Learning for NLP): 1,247 citations
- Paper 2 (RL in Robotics): 523 citations
- Paper 3 (Federated Learning): 312 citations
- Paper 4 (GNN Drug Discovery): 678 citations
- Paper 5 (Self-Supervised CV): 445 citations
- Paper 6 (LLM Code Generation): 892 citations
- Paper 7 (Efficient Transformers): 1,056 citations
- Paper 8 (Multimodal Learning): 634 citations
- Paper 9 (NAS): 389 citations
- Paper 10 (Trustworthy AI): 567 citations

Total citations across collection: 6,743
"""


async def main():
    # Load environment variables
    load_dotenv()

    # Build the RLM graph
    graph = create_openai_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
        temperature=0.7,
    )

    # Example queries to demonstrate different analysis patterns
    queries = [
        "Which paper has the most citations and what is it about?",
        "How many papers were published in 2024 and what topics do they cover?",
        "Find all papers related to transformers or attention mechanisms.",
        "What are the key findings about self-supervised learning from Paper 5?",
    ]

    print(f"Document length: {len(LONG_DOCUMENT):,} characters")
    print("=" * 60)

    # Run with the first query as demonstration
    query = queries[0]

    initial_state = create_initial_state(
        query=query,
        context=LONG_DOCUMENT,
        max_iterations=15,
    )

    config = {"configurable": {"thread_id": f"rlm-doc-{uuid.uuid4()}"}}

    print(f"\nQuery: {query}")
    print("-" * 60)

    result = await graph.ainvoke(initial_state, config)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result.get("final_answer", "No answer found"))
    print("=" * 60)
    print(f"\nIterations: {result.get('iteration', 0)}")
    print(f"Total LLM calls: {result.get('total_llm_calls', 0)}")

    # Optionally run additional queries
    print("\n" + "=" * 60)
    print("Additional example queries you can try:")
    for i, q in enumerate(queries[1:], 2):
        print(f"  {i}. {q}")


async def run_all_queries():
    """Run all example queries."""
    load_dotenv()

    graph = create_openai_rlm_graph(
        root_model="gpt-4o",
        recursive_model="gpt-4o-mini",
        temperature=0.7,
    )

    queries = [
        "Which paper has the most citations and what is it about?",
        "How many papers were published in 2024?",
        "Find all papers related to transformers.",
        "What are the key findings from Paper 5?",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("-" * 60)

        initial_state = create_initial_state(
            query=query,
            context=LONG_DOCUMENT,
            max_iterations=15,
        )

        config = {"configurable": {"thread_id": f"rlm-doc-{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config)

        print(f"\nAnswer: {result.get('final_answer', 'No answer found')}")
        print(f"Iterations: {result.get('iteration', 0)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        asyncio.run(run_all_queries())
    else:
        asyncio.run(main())
