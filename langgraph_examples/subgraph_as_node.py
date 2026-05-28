"""
LangGraph example demonstrating subgraphs.

A subgraph is a graph used as a node for another graph.

This example considers the invokation of a subgraph from within a node
of the parent graph. The state of the parent and child graphs may be
different.

Parent graph
"""

import operator
import time
from pathlib import Path
from typing import Annotated, TypedDict

import mlflow.langchain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from utils.utils import save_mermaid_png

mlflow.langchain.autolog()

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"
PARENT_THREAD_ID = "parent-thread"
DELAY = 10


class ParentState(TypedDict):
    """State for the parent graph."""

    text: Annotated[str, operator.add]
    word_count: int
    char_count: int
    uppercased: str
    reversed_text: str


# --- Subgraph 1: Text Analyzer ---


def count_words(state: ParentState) -> dict:
    """Count the number of words."""
    print("Counting words...")
    time.sleep(DELAY)
    return {"word_count": len(state["text"].split())}


def count_chars(state: ParentState) -> dict:
    """Count the number of characters."""
    print("Counting characters...")
    time.sleep(DELAY)
    return {"char_count": len(state["text"])}


def build_analyzer_subgraph() -> CompiledStateGraph:
    """Build and compile the text analyzer subgraph."""
    builder = StateGraph(ParentState)
    builder.add_node("count_words", count_words)
    builder.add_node("count_chars", count_chars)

    builder.add_edge(START, "count_words")
    builder.add_edge(START, "count_chars")
    builder.add_edge("count_words", END)
    builder.add_edge("count_chars", END)

    # Inherits checkpointer from parent graph
    return builder.compile(checkpointer=None)  # None is default value


# --- Subgraph 2: Text Transformer ---


def to_uppercase(state: ParentState) -> dict:
    """Convert text to uppercase."""
    print("Converting text to uppercase...")
    time.sleep(DELAY)
    return {"uppercased": state["text"].upper()}


def reverse_text(state: ParentState) -> dict:
    """Reverse the text."""
    print("Reversing text...")
    time.sleep(DELAY)
    return {"reversed_text": state["text"][::-1]}


def build_transformer_subgraph() -> CompiledStateGraph:
    """Build and compile the text transformer subgraph."""
    builder = StateGraph(ParentState)
    builder.add_node("to_uppercase", to_uppercase)
    builder.add_node("reverse_text", reverse_text)

    builder.add_edge(START, "to_uppercase")
    builder.add_edge(START, "reverse_text")
    builder.add_edge("to_uppercase", END)
    builder.add_edge("reverse_text", END)

    # Inherits checkpointer from parent graph
    return builder.compile(checkpointer=None)  # None is default value


# --- Parent Graph ---


def coordinator(state: ParentState) -> dict:
    """Call both subgraphs and merge their results into the parent state."""
    print("Coordinating subgraph invocations...")
    print(state)
    time.sleep(DELAY)
    return


def build_parent_graph(checkpointer: MemorySaver) -> CompiledStateGraph:
    """Build and compile the parent graph."""
    builder = StateGraph(ParentState)

    analyzer_subgraph = build_analyzer_subgraph()
    transformer_subgraph = build_transformer_subgraph()

    builder.add_node("coordinator", coordinator)
    builder.add_node("analyzer", analyzer_subgraph)
    builder.add_node("transformer", transformer_subgraph)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "analyzer")
    builder.add_edge("coordinator", "transformer")
    builder.add_edge("analyzer", END)
    builder.add_edge("transformer", END)
    return builder.compile(checkpointer)


def print_subgraph_histories(
    parent_graph: CompiledStateGraph,
    checkpointer: MemorySaver,
    thread_id: str,
) -> None:
    """Print state history for all subgraph namespaces in the checkpointer."""
    thread_storage = checkpointer.storage.get(thread_id, {})
    subgraph_namespaces = sorted(ns for ns in thread_storage if ns)

    for ns in subgraph_namespaces:
        subgraph_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
            }
        }
        subgraph_history = list(
            parent_graph.get_state_history(subgraph_config)
        )
        node_name = ns.split(":")[0] if ":" in ns else ns
        print(
            f"State history for subgraph '{node_name}'"
            f" (ns='{ns}', length): {len(subgraph_history)}"
        )
        for i, snapshot in enumerate(subgraph_history):
            print(
                f"  Checkpoint {i}: step={snapshot.metadata.get('step')}, "
                f"next={snapshot.next}"
            )
        print("=" * 50)


def run() -> None:
    """Run the example."""
    checkpointer = MemorySaver()

    parent_graph = build_parent_graph(checkpointer)
    save_mermaid_png(parent_graph)

    parent_config = {"configurable": {"thread_id": PARENT_THREAD_ID}}

    result = parent_graph.invoke(
        {"text": "Hello world from LangGraph subgraphs"},
        config=parent_config,
    )

    print(f"Input:      {result['text']}")
    print(f"Words:      {result['word_count']}")
    print(f"Chars:      {result['char_count']}")
    print(f"Uppercase:  {result['uppercased']}")
    print(f"Reversed:   {result['reversed_text']}")
    print("=" * 50)

    history = list(parent_graph.get_state_history(parent_config))
    print("State history Parent graph (length):", len(history))
    for i, snapshot in enumerate(history):
        print(
            f"Checkpoint {i}: step={snapshot.metadata.get('step')}, "
            f"next={snapshot.next}"
        )
    print("=" * 50)

    print_subgraph_histories(parent_graph, checkpointer, PARENT_THREAD_ID)


if __name__ == "__main__":
    run()
