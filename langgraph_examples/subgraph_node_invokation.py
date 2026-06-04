"""
LangGraph example demonstrating subgraphs.

A subgraph is a graph used as a node for another graph.

This example considers the invokation of a subgraph from within a node
of the parent graph. The state of the parent and child graphs may be
different.

Parent graph
"""

import operator
from typing import Annotated, TypedDict

import mlflow.langchain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from utils.utils import save_mermaid_png

mlflow.langchain.autolog()

PARENT_THREAD_ID = "parent-thread"
ANALYZER_THREAD_ID = "analyzer-thread"
TRANSFORMER_THREAD_ID = "transformer-thread"


# --- Runtime subgraph context ---


class SubgraphContext(TypedDict):
    """Context for the subgraph invokation example."""

    analyzer_graph: CompiledStateGraph
    transformer_graph: CompiledStateGraph
    analyzer_config: dict
    transformer_config: dict


# --- Subgraph 1: Text Analyzer ---


class AnalyzerState(TypedDict):
    """State for the text analyzer subgraph."""

    text: str
    word_count: int
    char_count: int


def count_words(state: AnalyzerState) -> dict:
    """Count the number of words."""
    return {"word_count": len(state["text"].split())}


def count_chars(state: AnalyzerState) -> dict:
    """Count the number of characters."""
    return {"char_count": len(state["text"])}


def build_analyzer_subgraph(checkpointer: MemorySaver) -> CompiledStateGraph:
    """Build and compile the text analyzer subgraph."""
    builder = StateGraph(AnalyzerState)
    builder.add_node("count_words", count_words)
    builder.add_node("count_chars", count_chars)

    builder.add_edge(START, "count_words")
    builder.add_edge(START, "count_chars")
    builder.add_edge("count_words", END)
    builder.add_edge("count_chars", END)
    return builder.compile(checkpointer=checkpointer)


# --- Subgraph 2: Text Transformer ---


class TransformerState(TypedDict):
    """State for the text transformer subgraph."""

    text: str
    uppercased: str
    reversed_text: str


def to_uppercase(state: TransformerState) -> dict:
    """Convert text to uppercase."""
    return {"uppercased": state["text"].upper()}


def reverse_text(state: TransformerState) -> dict:
    """Reverse the text."""
    return {"reversed_text": state["text"][::-1]}


def build_transformer_subgraph(
    checkpointer: MemorySaver,
) -> CompiledStateGraph:
    """Build and compile the text transformer subgraph."""
    builder = StateGraph(TransformerState)
    builder.add_node("to_uppercase", to_uppercase)
    builder.add_node("reverse_text", reverse_text)

    builder.add_edge(START, "to_uppercase")
    builder.add_edge(START, "reverse_text")
    builder.add_edge("to_uppercase", END)
    builder.add_edge("reverse_text", END)
    return builder.compile(checkpointer=checkpointer)


# --- Parent Graph ---


class ParentState(TypedDict):
    """State for the parent graph."""

    text: str
    outputs: Annotated[list[str | int], operator.add]


def coordinator(state: ParentState, runtime: Runtime[SubgraphContext]) -> dict:
    """Call both subgraphs and merge their results into the parent state."""
    analyzer = runtime.context["analyzer_graph"]
    transformer = runtime.context["transformer_graph"]
    analyzer_config = runtime.context["analyzer_config"]
    transformer_config = runtime.context["transformer_config"]

    # Invokes each subgraph with its own state and thread
    analysis = analyzer.invoke({"text": state["text"]}, config=analyzer_config)
    transformation = transformer.invoke(
        {"text": state["text"]}, config=transformer_config
    )

    return {
        "outputs": [
            analysis["word_count"],
            analysis["char_count"],
            transformation["uppercased"],
            transformation["reversed_text"],
        ]
    }


def build_graph(checkpointer: MemorySaver) -> CompiledStateGraph:
    """Build and compile the parent graph."""
    builder = StateGraph(ParentState, context_schema=SubgraphContext)
    builder.add_node("coordinator", coordinator)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", END)
    return builder.compile(checkpointer)


def print_subgraph_history(
    graph: CompiledStateGraph,
    name: str,
    config: dict,
) -> None:
    """Print state history for a subgraph."""
    history = list(graph.get_state_history(config))
    print(f"State history for subgraph '{name}' (length): {len(history)}")
    for i, snapshot in enumerate(history):
        print(
            f"  Checkpoint {i}: step={snapshot.metadata.get('step')}, "
            f"next={snapshot.next}"
        )
    print("=" * 50)


def run() -> None:
    """Run the example."""
    checkpointer = MemorySaver()

    analyzer_subgraph = build_analyzer_subgraph(checkpointer)
    transformer_subgraph = build_transformer_subgraph(checkpointer)

    parent_graph = build_graph(checkpointer)
    save_mermaid_png(parent_graph, __file__)

    # Different threads are important because they do not share state
    parent_config = {"configurable": {"thread_id": PARENT_THREAD_ID}}
    analyzer_config = {"configurable": {"thread_id": ANALYZER_THREAD_ID}}
    transformer_config = {"configurable": {"thread_id": TRANSFORMER_THREAD_ID}}

    result = parent_graph.invoke(
        {"text": "Hello world from LangGraph subgraphs"},
        config=parent_config,
        context={
            "analyzer_graph": analyzer_subgraph,
            "transformer_graph": transformer_subgraph,
            "analyzer_config": analyzer_config,
            "transformer_config": transformer_config,
        },
    )

    print(f"Input:      {result['text']}")
    print(f"Outputs:    {result['outputs']}")
    print("=" * 50)

    history = list(parent_graph.get_state_history(parent_config))
    print("State history Parent graph (length):", len(history))
    for i, snapshot in enumerate(history):
        print(
            f"Checkpoint {i}: step={snapshot.metadata.get('step')}, "
            f"next={snapshot.next}"
        )
    print("=" * 50)

    print_subgraph_history(analyzer_subgraph, "analyzer", analyzer_config)
    print_subgraph_history(
        transformer_subgraph, "transformer", transformer_config
    )


if __name__ == "__main__":
    run()
