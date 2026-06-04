"""
Define reducer function to handle complex state.

The graph will run in parallel three review nodes that must be
correctly appended to the state with the custom reducer function.

{
    "reviews": {
        "books": [
            {
                "id": "book_id",
                "name": "book1",
                "review": "This book was great!"
            }
        ],
        "movies": [
            {
                "id": "movie_id",
                "name": "movie1",
                "review": "This movie was terrible."
            }
        ],
        "songs": [
            {
                "id": "song_id",
                "name": "song1",
                "review": "The melody was fantastic."
            }
        ]
    }
}

"""

from typing import Annotated

import mlflow.langchain
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from config.llm_model import LLM_MODEL
from utils.utils import save_mermaid_png

mlflow.langchain.autolog()

THREAD_ID = "demo-thread-1"


class Review(BaseModel):
    """Model for a review item."""

    id: str
    name: str
    review: str


class ReviewsChannel(BaseModel):
    """Model for the reviews channel."""

    books: list[Review] = []
    movies: list[Review] = []
    songs: list[Review] = []


def handle_reviews(
    current: ReviewsChannel, update: ReviewsChannel | dict[str, Review]
) -> dict:
    """Reducer function to handle reviews channel."""
    if not update:
        return current

    print("=" * 50)
    print("Current:", current)
    print("Update:", update)
    print("=" * 50)

    if isinstance(update, dict):
        update = ReviewsChannel(**update)

    return ReviewsChannel(
        books=current.books + update.books,
        movies=current.movies + update.movies,
        songs=current.songs + update.songs,
    )


class GraphState(TypedDict):
    """Short-term memory state of the graph. Encodes dynamic memory."""

    messages: Annotated[list, add_messages]
    reviews: Annotated[ReviewsChannel, handle_reviews]


def coordinator(state: GraphState) -> dict:
    """Process user input to determine which review node to invoke."""
    llm = LLM_MODEL

    system_message = SystemMessage(
        "You coordinate review of books, movies, and songs.",
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        response = llm.invoke(messages)
    else:
        response = llm.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [response]}


@tool
def book_review(
    book_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Review a book."""
    review = Review(
        id="book_id",
        name=book_name,
        review="This book was great!",
    )
    review_update = ReviewsChannel(books=[review])

    message_update = ToolMessage(
        "Book reviewed successfully.", tool_call_id=tool_call_id
    )
    return Command(
        update={
            "reviews": review_update,
            "messages": [message_update],
        }
    )


@tool
def movie_review(
    movie_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Review a movie."""
    review = Review(
        id="movie_id",
        name=movie_name,
        review="This movie was terrible.",
    )
    review_update = ReviewsChannel(movies=[review])

    message_update = ToolMessage(
        "Movie reviewed successfully.", tool_call_id=tool_call_id
    )
    return Command(
        update={
            "reviews": review_update,
            "messages": [message_update],
        }
    )


@tool
def song_review(
    song_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Review a song."""
    review = Review(
        id="song_id",
        name=song_name,
        review="The melody was fantastic.",
    )
    update_song_review = {"songs": [review]}

    message_update = ToolMessage(
        "Song reviewed successfully.", tool_call_id=tool_call_id
    )
    return Command(
        update={
            "reviews": update_song_review,
            "messages": [message_update],
        }
    )


TOOLS = [book_review, movie_review, song_review]


def build_graph(checkpointer: MemorySaver) -> CompiledStateGraph:
    """Build and compile the graph."""
    builder = StateGraph(GraphState)

    builder.add_node("coordinator", coordinator)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges("coordinator", tools_condition)
    builder.add_edge("tools", "coordinator")
    builder.add_edge("coordinator", END)
    return builder.compile(checkpointer=checkpointer)


def run() -> None:
    """Run the example."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)
    save_mermaid_png(graph, __file__)

    config = {"configurable": {"thread_id": THREAD_ID}}

    graph_result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    "Review the book The Lord of The Rings "
                    "the song Bohemian Rhapsody and "
                    "the movie Inception."
                )
            ]
        },
        config=config,
    )

    # StateSnapshot — persisted state of the thread after invocation
    state_snapshot = graph.get_state(config)
    print("StateSnapshot values:", state_snapshot.values)
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

    print("Last message:", graph_result["messages"][-1])
    print("=" * 50)
    print("Reviews channel value:", graph_result["reviews"])
    print("=" * 50)


if __name__ == "__main__":
    run()
