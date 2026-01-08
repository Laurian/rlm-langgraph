"""LangGraph server module for langgraph dev.

This module exports a pre-configured RLM graph for use with `langgraph dev`.

Usage:
    langgraph dev

Then open http://localhost:8000 in your browser.

Example input:
{
    "query": "What is the total price of all in-stock products?",
    "context": "Product Catalog:\\n1. Widget Pro - $49.99 (In stock: Yes)\\n2. Gadget Plus - $79.99 (In stock: No)\\n3. Super Tool - $129.99 (In stock: Yes)"
}
"""

from dotenv import load_dotenv

from rlm_langgraph.graph import create_openai_rlm_graph

# Load environment variables
load_dotenv()

# Export graph for langgraph dev
# Using gpt-4o-mini for fast iteration during development
# checkpointer=None because LangGraph API handles persistence automatically
graph = create_openai_rlm_graph(
    root_model="gpt-4o-mini",
    recursive_model="gpt-4o-mini",
    temperature=0.0,
    checkpointer=None,
)
