"""Utility functions used across the repository."""

from pathlib import Path

from langgraph.graph.state import CompiledStateGraph


def save_mermaid_png(graph: CompiledStateGraph, caller_file: str) -> None:
    """Save compiled graph as a PNG file next to the calling script."""
    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    path = Path(caller_file).parent / "latest_graph_run.png"
    path.write_bytes(png_data)
