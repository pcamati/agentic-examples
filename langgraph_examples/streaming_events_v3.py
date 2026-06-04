"""
Simple agentic streaming events in LangGraph.

Demonstrates the v3 streaming protocol, which emits ProtocolEvent dicts shaped:

    {
        "type":   "event",
        "method": str,       # stream mode: "values", "messages", "custom",
                             #              "updates", "lifecycle"
        "seq":    int,       # monotonic ordering number
        "params": {
            "namespace": list[str],  # graph path ([] for root)
            "timestamp": int,        # wall-clock ms
            "data":      Any,        # varies by method — see stream_events()
        },
    }

By default v3 registers ValuesTransformer, MessagesTransformer,
LifecycleTransformer, and SubgraphTransformer.  We add CustomTransformer and
UpdatesTransformer via the `transformers=` argument to also surface "custom"
and "updates" methods.
"""

import mlflow.langchain
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.stream.transformers import CustomTransformer, UpdatesTransformer

from config.llm_model import LLM_MODEL
from utils.utils import save_mermaid_png

mlflow.langchain.autolog()


@tool
def get_current_temperature() -> int:
    """Get the current temperature for a given location."""
    writer = get_stream_writer()
    writer("WE ARE INSIDE THE TOOL")
    return 22


TOOLS = [get_current_temperature]


def llm_node(state: MessagesState) -> dict:
    """Call the LLM with the current message state and return the response."""
    llm = LLM_MODEL
    writer = get_stream_writer()

    # Emit a custom message visible on the "custom" stream method
    writer("WE ARE INSIDE THE LLM NODE")

    last_message = state["messages"][-1]
    # After tool results are returned, invoke without tools so the model is
    # forced to produce a natural-language reply instead of another tool call.
    if isinstance(last_message, ToolMessage):
        response = llm.invoke(state["messages"])
    else:
        response = llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(MessagesState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    builder.add_edge("llm", END)

    return builder.compile()


def _sep(label: str, seq: object) -> None:
    """Print a separator line with the event type and sequence number."""
    print(f"\n{'─' * 50}")
    print(f"  [{seq}] {label}")
    print("─" * 50)


def _handle_custom(data: object, namespace: list, seq: object) -> None:
    """Print a custom event emitted via get_stream_writer()."""
    _sep("CUSTOM", seq)
    print(f"  namespace : {namespace or '(root)'}")
    print(f"  payload   : {data!r}")


def _handle_messages_token(payload: dict, node: str, seq: object) -> None:
    """Print a single streaming token chunk from an LLM."""
    text = ""
    if payload.get("event") == "content-block-delta":
        text = payload.get("delta", {}).get("text", "")
    if text:
        _sep("TOKEN (streaming chunk)", seq)
        print(f"  node  : {node}")
        print(f"  token : {text!r}")


def _handle_messages_full(payload: object, node: str, seq: object) -> None:
    """Print a complete AIMessage returned by .invoke()."""
    has_tool_calls = bool(getattr(payload, "tool_calls", None))
    _sep("MESSAGE (full output)", seq)
    print(f"  node : {node}")
    if has_tool_calls:
        calls = [
            f"{tc['name']}({tc['args']})"
            for tc in payload.tool_calls  # type: ignore[union-attr]
        ]
        print(f"  tool_calls : {calls}")
    else:
        print(f"  content : {payload.content[:300]}")  # type: ignore[union-attr]


def _handle_messages(data: object, _namespace: list, seq: object) -> None:
    """
    Dispatch a messages event to the right sub-handler.

    data = (payload, metadata); metadata carries "langgraph_node".
    payload is either a dict with "event" key (streaming token) or a
    whole BaseMessage (when the model was called with .invoke()).
    """
    payload, metadata = data if isinstance(data, tuple) else (data, {})
    node = (
        metadata.get("langgraph_node", "?")
        if isinstance(metadata, dict)
        else "?"
    )
    if isinstance(payload, dict) and "event" in payload:
        _handle_messages_token(payload, node, seq)
    elif (
        hasattr(payload, "content")
        and not isinstance(payload, AIMessageChunk)
        and not isinstance(payload, ToolMessage)
    ):
        _handle_messages_full(payload, node, seq)


def _handle_updates(data: object, namespace: list, seq: object) -> None:
    """
    Print which state keys changed after a node completed.

    data = {node_name: {state_key: new_value, ...}}
    """
    if not isinstance(data, dict):
        return
    _sep("UPDATE (node state diff)", seq)
    print(f"  namespace : {namespace or '(root)'}")
    for node_name, update in data.items():
        if isinstance(update, dict):
            summary = {
                k: (
                    f"list[{len(v)}]"
                    if isinstance(v, list)
                    else type(v).__name__
                )
                for k, v in update.items()
            }
            print(f"  {node_name} → {summary}")
        else:
            print(f"  {node_name} → {type(update).__name__}")


def _handle_lifecycle(data: object, _namespace: list, seq: object) -> None:
    """
    Print a subgraph lifecycle event (start / finish).

    data = LifecyclePayload: {"event": status, "namespace": [...], ...}
    Particularly useful when the graph contains nested subgraphs.
    """
    if not isinstance(data, dict):
        return
    _sep("LIFECYCLE (subgraph event)", seq)
    print(f"  event      : {data.get('event')}")
    print(f"  namespace  : {data.get('namespace', [])}")
    print(f"  graph_name : {data.get('graph_name', '')!r}")


def stream_events(
    state: MessagesState, graph: CompiledStateGraph
) -> dict | None:
    """
    Stream and print v3 protocol events from the graph.

    Each ProtocolEvent carries a `method` that identifies what kind of data
    arrived and a `params.data` whose shape depends on that method:

    - "custom"    data = anything passed to get_stream_writer() inside a node
    - "messages"  data = (payload, metadata)
    - "updates"   data = {node_name: state_update_dict}
    - "lifecycle" data = LifecyclePayload (subgraph start/finish)
    - "values"    data = full MessagesState snapshot after each step

    Returns the final state snapshot (last "values" event).
    """
    last_values: dict | None = None
    _handlers = {
        "custom": _handle_custom,
        "messages": _handle_messages,
        "updates": _handle_updates,
        "lifecycle": _handle_lifecycle,
    }

    for event in graph.stream_events(
        state,
        version="v3",
        # CustomTransformer  → surfaces get_stream_writer() output
        # UpdatesTransformer → surfaces per-node state diffs
        transformers=[CustomTransformer, UpdatesTransformer],
    ):
        method = event.get("method")
        params = event.get("params", {})
        data = params.get("data")
        namespace = params.get("namespace", [])
        seq = event.get("seq", "?")

        if method == "values":
            last_values = data
        elif method in _handlers:
            _handlers[method](data, namespace, seq)

    return last_values


def run() -> None:
    """Run the example."""
    graph = build_graph()

    save_mermaid_png(graph, __file__)

    system_message = SystemMessage(
        "You are a helpful climate assistant. Respond the user query."
    )
    human_message = HumanMessage("What is the current temperature in Paris?")
    initial_state: MessagesState = {
        "messages": [system_message, human_message],
    }

    last_values = stream_events(initial_state, graph)

    print(f"\n{'═' * 50}")
    print("  FINAL STATE")
    print("═" * 50)
    if last_values:
        messages = last_values.get("messages", [])
        print(f"  total messages : {len(messages)}")
        last_msg = messages[-1] if messages else None
        if last_msg:
            print(f"  last type      : {type(last_msg).__name__}")
            print(f"  last content   : {last_msg.content[:200]}")
    else:
        print("  (no values event captured)")


if __name__ == "__main__":
    run()
