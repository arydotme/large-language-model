from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn

app = FastAPI(
    title='LangChain Server',
    version='1.0',
    description='Simple api server using Langchain'
)

model = ChatOllama(model = 'llama3.2:1b')

prompt = ChatPromptTemplate.from_template("give me a essay about { topic }")

add_routes(
    app,
    prompt | model,
    path="/essay"
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)