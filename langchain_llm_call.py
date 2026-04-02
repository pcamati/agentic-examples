"""Simples LLM call using Langchain and Ollama."""

from langchain_ollama import ChatOllama


def run() -> None:
    """Run the example."""
    llm = ChatOllama(model="llama3.2")
    llm_response = llm.invoke("What is an AI agent?")
    print(llm_response.content)


if __name__ == "__main__":
    run()
