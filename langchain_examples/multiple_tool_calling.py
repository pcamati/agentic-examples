"""Multiple tool calls using Langchain and Ollama."""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_current_temperature(location: str) -> str:
    """Get the current temperature for a given location."""
    return f"The current temperature in {location} is 22°C."


def run() -> None:
    """Run the example."""
    llm = ChatOllama(model="llama3.1:8b")
    llm_with_tools = llm.bind_tools([get_current_temperature])

    # Providing two cities to demonstrate multiple tool calls in one response
    human_message = HumanMessage(
        "What is the current temperature in Paris and New York?"
    )
    response = llm_with_tools.invoke([human_message])

    # Total number of tool calls batched by the model in a single response;
    # with a single-city prompt this would be 1 — here it should be 2
    print("Total tool calls:", len(response.tool_calls))
    print("=" * 50)

    # All tool call IDs issued in this response; each ID is unique and links
    # a specific ToolMessage result back to the correct request
    print("All tool call IDs:", [tc["id"] for tc in response.tool_calls])
    print("=" * 50)

    # Names of the tools the model chose to call; useful when multiple
    # different tools are bound and you want to see which were selected
    print("Tools invoked:", [tc["name"] for tc in response.tool_calls])
    print("=" * 50)

    tool_messages = []
    for i, tool_call in enumerate(response.tool_calls):
        # Position of this call within the batch (0-based)
        print(f"Tool call index: {i} of {len(response.tool_calls) - 1}")
        # Structured arguments the model extracted from natural language
        print(f"Tool call arguments: {tool_call['args']}")
        print("=" * 50)

        tool_message: ToolMessage = get_current_temperature.invoke(tool_call)
        tool_messages.append(tool_message)

        # ToolMessage attributes that flow back into the conversation
        print(f"ToolMessage type: {tool_message.type}")
        print(f"ToolMessage tool_call_id: {tool_message.tool_call_id}")
        print(f"ToolMessage content: {tool_message.content}")
        print("=" * 50)

    # All tool call IDs must be represented in the ToolMessages sent back;
    # a mismatch here would cause the model to error or ignore results
    response_ids = {tc["id"] for tc in response.tool_calls}
    message_ids = {tm.tool_call_id for tm in tool_messages}
    print("Tool call IDs matched:", response_ids == message_ids)
    print("=" * 50)

    # Full agentic loop: send AIMessage + all ToolMessages back so the model
    # can synthesise a single answer covering every location
    final_response = llm_with_tools.invoke(
        [human_message, response, *tool_messages]
    )
    print("Final answer:", final_response.content)
    print("=" * 50)


if __name__ == "__main__":
    run()
