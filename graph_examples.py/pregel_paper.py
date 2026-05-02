"""
Reproduce the graph example presented in the Pregel paper.

Link: https://doi.org/10.1145/1807167.1807184
"""

import time
from typing import TYPE_CHECKING, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

SLEEP_TIME = 5


class GraphState(TypedDict):
    """State of the graph."""

    node_a: int
    node_b: int
    node_c: int
    node_d: int


def node_a(state: GraphState) -> dict:
    """Node logic."""
    print("State Node A: ", state["node_a"])
    time.sleep(SLEEP_TIME)
    next_nodes = ["node_B"]
    largest = max([state["node_b"]])
    if largest > state["node_a"]:
        update_state = {"node_a": largest}
        return Command(update=update_state, goto=next_nodes)
    return {}


def node_b(state: GraphState) -> dict:
    """Node logic."""
    print("State Node B: ", state["node_b"])
    time.sleep(SLEEP_TIME)
    next_nodes = ["node_A", "node_D"]
    largest = max([state["node_b"], state["node_c"]])
    if largest > state["node_b"]:
        update_state = {"node_b": largest}
        return Command(update=update_state, goto=next_nodes)
    return {}


def node_c(state: GraphState) -> dict:
    """Node logic."""
    print("State Node C: ", state["node_c"])
    time.sleep(SLEEP_TIME)
    next_nodes = ["node_B"]
    largest = max([state["node_d"]])
    if largest > state["node_c"]:
        update_state = {"node_c": largest}
        return Command(update=update_state, goto=next_nodes)
    return {}


def node_d(state: GraphState) -> dict:
    """Node logic."""
    print("State Node D: ", state["node_d"])
    time.sleep(SLEEP_TIME)
    next_nodes = ["node_C"]
    largest = max([state["node_b"], state["node_c"]])
    if largest > state["node_d"]:
        update_state = {"node_d": largest}
        return Command(update=update_state, goto=next_nodes)
    return {}


def run() -> None:
    """Run the example."""
    builder = StateGraph(GraphState)

    builder.add_node("node_A", node_a)
    builder.add_node("node_B", node_b)
    builder.add_node("node_C", node_c)
    builder.add_node("node_D", node_d)

    builder.add_edge(START, "node_A")
    builder.add_edge(START, "node_B")
    builder.add_edge(START, "node_C")
    builder.add_edge(START, "node_D")

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    initial_state = {
        "node_a": 3,
        "node_b": 6,
        "node_c": 2,
        "node_d": 1,
    }
    # initial_state = {
    #     "node_a": 10,
    #     "node_b": 6,
    #     "node_c": 20,
    #     "node_d": 30,
    # }
    graph.invoke(
        initial_state,
        config=config,
    )

    history = list(graph.get_state_history(config))
    for step, snapshot in enumerate(history[::-1]):
        print(
            f"Superstep {step}: state={snapshot.values}, next={snapshot.next}"
        )


if __name__ == "__main__":
    run()
