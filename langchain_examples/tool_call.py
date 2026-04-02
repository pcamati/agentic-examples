"""Tool call using Langchain and Ollama."""

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

    # JSON schema the LLM receives to understand how to call the tool
    print(
        "Tool schema:", get_current_temperature.args_schema.model_json_schema()
    )
    print("=" * 50)

    # Full tool payload sent to the model (as registered on the bound LLM)
    print("Bound tools:", llm_with_tools.kwargs["tools"])
    print("=" * 50)

    human_message = HumanMessage("What is the current temperature in Paris?")
    response = llm_with_tools.invoke([human_message])

    # Text content from the model (usually empty when tools are called)
    print("Content:", response.content)
    print("=" * 50)

    # Structured tool call requests made by the model
    print("Tool calls:", response.tool_calls)
    print("=" * 50)

    # Provider-specific metadata (e.g. model name, stop reason, timing)
    print("Response metadata:", response.response_metadata)
    print("=" * 50)

    # Token counts for the prompt, completion, and total
    print("Usage metadata:", response.usage_metadata)
    print("=" * 50)

    # Execute the tool calls and inspect the resulting ToolMessage
    tool_messages = []
    for tool_call in response.tool_calls:
        # Unique ID used to link this result back to the model's request
        print(f"Tool call ID: {tool_call['id']}")
        # Structured arguments the model extracted from natural language
        print(f"Tool call arguments: {tool_call['args']}")
        print("=" * 50)

        tool_message: ToolMessage = get_current_temperature.invoke(tool_call)
        tool_messages.append(tool_message)

        # Alternative: invoke with args only and construct ToolMessage manually
        result = get_current_temperature.invoke(tool_call["args"])
        tool_message_manual = ToolMessage(
            content=result, tool_call_id=tool_call["id"]
        )

        # ToolMessage attributes that flow back into the conversation
        print(f"ToolMessage type: {tool_message.type}")
        print(f"ToolMessage tool_call_id: {tool_message.tool_call_id}")
        print(f"ToolMessage content: {tool_message.content}")
        print(f"ToolMessage manual content: {tool_message_manual.content}")
        print("=" * 50)

    # Full agentic loop: send AIMessage + ToolMessages back to get the
    # final answer
    final_response = llm_with_tools.invoke(
        [human_message, response, *tool_messages]
    )
    print("Final answer:", final_response.content)
    print("=" * 50)


if __name__ == "__main__":
    run()
