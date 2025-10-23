'''
Learn to create Large Language Model (LLM)

This LLM using LLaMA 3.2 and LangSmith for monitoring
'''

# Import library
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = "True"
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are helpful assistant. Please response to the user queries'),
        ('user', 'Question: {question}')
    ]
)

# Streamlit framework
st.title('Langchain with LLaMA 3.2')
input_text = st.text_input("What topic u want search")

# Ollama LLaMA LLM
llm = OllamaLLM(model = 'llama3.2:1b')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({ 'question': input_text }))