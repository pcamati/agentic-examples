"""Configure the LLM model instance to be used across all examples."""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# from langchain_openai import ChatOpenAI

load_dotenv()


LLM_MODEL = ChatOllama(model="llama3.1:8b")
# LLM_MODEL = ChatOpenAI(model="gpt-5.4")

if __name__ == "__main__":
    # Test LLM model configuration
    response = LLM_MODEL.invoke("What is the capital of France?")
    print(response.content)
