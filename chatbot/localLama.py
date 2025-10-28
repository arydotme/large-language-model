"""
Learn to create Large Language Model (LLM)

This LLM using LLaMA 3.2:1b and LangSmith for monitoring
"""

# Import library
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.langchain.org")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the user's request only based on the given context.",
        ),
        ("user", "Question: {question}"),
    ]
)

# Streamlit framework
st.title("Langchain with LLaMA 3.2")
input_text = st.text_input("What topic u want search")

# Ollama LLaMA LLM
model = ChatOllama(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

"""
Use the traceable decorator to trace entire application instead of just the LLM calls

The @traceable decorator is a simple way to log traces from the LangSmith Python SDK
"""


@traceable(run_type="chain", name="local-llama-chat")
def run_chain(question):
    return chain.invoke({"question": question})


if input_text:
    st.write(run_chain(input_text))
