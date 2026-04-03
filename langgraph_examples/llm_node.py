"""Simple LangGraph graph with a single LLM node using Ollama."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

GRAPH_PNG_PATH = Path(__file__).parent / "latest_graph_run.png"
THREAD_ID = "demo-thread-1"


@dataclass
class Context:
    """Static per-invocation data injected via Runtime."""

    database_connection: str  # Emulates the persistent database connection
    user_id: str  # Caller identity — useful for personalisation or audit


class GraphState(TypedDict):
    """Short-term memory state of the graph. Encodes dynamic memory."""

    messages: Annotated[
        list, add_messages
    ]  # Reducer merges new messages into the list
    topic: str  # Extra channel to track the conversation subject


def llm_node(state: GraphState, runtime: Runtime[Context]) -> dict:
    """Call the LLM with the current message state and return the response."""
    context = runtime.context
    database_connection = context.database_connection
    if not database_connection:
        error_msg = "Database connection not loaded."
        raise ValueError(error_msg)
    llm = ChatOllama(model="llama3.1:8b")
    messages = [
        SystemMessage(f"Use this user_id: {context.user_id}"),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph(checkpointer: MemorySaver) -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(GraphState, context_schema=Context)
    builder.add_node("llm", llm_node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile(checkpointer=checkpointer)


def run() -> None:
    """Run the example."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    # --- CompiledStateGraph ---
    print("Graph type:", type(graph))
    print("=" * 50)

    # Nodes and edges registered in the graph
    drawable = graph.get_graph()
    print("Graph nodes:", list(drawable.nodes.keys()))
    print("Graph edges:", [(e.source, e.target) for e in drawable.edges])
    print("=" * 50)

    # Checkpointer — persists thread state between invocations
    # (enables multi-turn conversations)
    print("Checkpointer:", graph.checkpointer)
    print("Checkpointer type:", type(graph.checkpointer))
    print("=" * 50)

    # Store — optional external key-value store for cross-thread
    # long-term memory (None if not set)
    print("Store:", graph.store)
    print("=" * 50)

    # Input/output channels — state keys consumed as input and
    # exposed in the output
    print("Input channels:", graph.input_channels)
    print("Output channels:", graph.output_channels)
    print("=" * 50)

    # Config specs — runtime-configurable fields
    # (e.g. thread_id injected by the checkpointer)
    print("Config specs:", graph.config_specs)
    print("=" * 50)

    # JSON schemas derived from the TypedDict state definition
    print("Input JSON schema:", graph.get_input_jsonschema())
    print("=" * 50)
    print("Output JSON schema:", graph.get_output_jsonschema())
    print("=" * 50)

    # context_schema — the dataclass/TypedDict/BaseModel registered for
    # Context injection
    print("Context schema:", graph.context_schema)
    print("=" * 50)

    # Context JSON schema — auto-derived from the Context dataclass fields
    print("Context JSON schema:", graph.get_context_jsonschema())
    print("=" * 50)

    png_data = drawable.draw_mermaid_png()
    GRAPH_PNG_PATH.write_bytes(png_data)

    context = Context(database_connection="connection_here", user_id="alice")

    graph_result = graph.invoke(
        {"messages": [HumanMessage("What is Physics?")], "topic": "science"},
        config=config,
        # context is passed as a direct kwarg — NOT inside configurable
        # LangGraph wraps it in a Runtime and injects it into every node
        context=context,
    )

    # graph_result is a GraphState dict — the graph's final state snapshot
    print("Graph result type:", type(graph_result))
    print("Graph result keys:", list(graph_result.keys()))
    print("Topic channel value:", graph_result["topic"])
    print("=" * 50)

    # StateSnapshot — persisted state of the thread after invocation
    state_snapshot = graph.get_state(config)
    print("StateSnapshot type:", type(state_snapshot))
    print("StateSnapshot values:", state_snapshot.values)
    print("StateSnapshot next nodes:", state_snapshot.next)
    print("StateSnapshot config:", state_snapshot.config)
    print("StateSnapshot metadata:", state_snapshot.metadata)
    print("StateSnapshot created at:", state_snapshot.created_at)
    print("=" * 50)

    # State history — all checkpoints saved for this thread (most recent first)
    history = list(graph.get_state_history(config))
    print("State history length:", len(history))
    for i, snapshot in enumerate(history):
        print(
            f"Checkpoint {i}: step={snapshot.metadata.get('step')}, "
            f"next={snapshot.next}"
        )
    print("=" * 50)


if __name__ == "__main__":
    run()
