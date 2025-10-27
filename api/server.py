from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI(
    title='Langchain Server',
    version='1.0',
    description='Simple api server using Langchain'
)

model = ChatOllama(model = 'llama3.2:1b')

add_routes(
    app,
    model,
    path='/ollama'
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)