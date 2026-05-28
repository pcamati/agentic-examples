"""Utility functions used across the repository."""

from langgraph.graph.state import CompiledStateGraph

from langgraph_examples.context_usage import GRAPH_PNG_PATH


def save_mermaid_png(graph: CompiledStateGraph) -> None:
    """Save compiled graph as a PNG file."""
    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)
