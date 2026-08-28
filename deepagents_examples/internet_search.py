"""Deepagents agent with a built-in web search tool."""

import mlflow.langchain
from deepagents import create_deep_agent

from config.llm_model import LLM_MODEL

mlflow.langchain.autolog()

# OpenAI's built-in web search — no extra install or API key needed
internet_search = {"type": "web_search"}

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model=LLM_MODEL,
    tools=[internet_search],
    system_prompt=research_instructions,
)


def run() -> None:
    """Run the example."""
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Prove you have internet access."}
            ]
        }
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    run()
