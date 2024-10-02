import os

import langchain
import numpy as np
import openai
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    Tool,
    create_openai_tools_agent,
    create_react_agent,
    tool,
)
from langchain.chains import LLMMathChain, RetrievalQA
from langchain.memory import ConversationEntityMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT_VERSION = "2024-06-01"
OPENAI_MODEL_NAME = "gpt-4o"

OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

llm = AzureChatOpenAI(
    deployment_name=OPENAI_MODEL_NAME,
    model_name=OPENAI_MODEL_NAME,
    azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
    api_version=OPENAI_DEPLOYMENT_VERSION,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    deployment=OPENAI_ADA_EMBEDDING_MODEL_NAME,
    model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
    chunk_size=1,
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_DEPLOYMENT_VERSION,
)


file_path = ".\data\BonBon FAQ.pdf"
loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(document)

persist_directory = "local_vectorstore"
local_store = "local_docstore"
collection_name = "react_agent"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# This text splitter is used to create the child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    persist_directory=os.path.join(PROJECT_ROOT, "data", persist_directory),
    collection_name=collection_name,
    embedding_function=embeddings,
)
# The storage layer for the parent documents
local_store = LocalFileStore(os.path.join(PROJECT_ROOT, "data", local_store))
store = create_kv_docstore(local_store)
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# # run only once
# vectorstore.persist()
# retriever.add_documents(documents, ids=None)

# Setting up ReAct Agent

doc_prompt = PromptTemplate.from_template(
    "<context>\n{page_content}\n\n<meta>\nsource: {source}\npage: {page} + 1\n</meta>\n</context>"
)
tool_search = create_retriever_tool(
    retriever=retriever,
    name="Search BonBon",
    description="Searches and returns answer from BONBON FAQ.",
    document_prompt=doc_prompt,
)

math_chain = LLMMathChain.from_llm(llm=llm)
duckduck = DuckDuckGoSearchRun()

tools = [
    tool_search,
    Tool(
        name="Duck Duck Go",
        func=duckduck.run,
        description="useful for when you need to search for more information on the internet with Duck Duck Go search",
    ),
    Tool(
        name="Calculator",
        func=math_chain.run,
        description="use this tool for math calculating",
    ),
]

template = """
Answer the following questions as best you can. You can use history {history} to fill in unknown context. You have access to the following tools:

{tools}

Use the following format:
Remember: If any invalid format occurs, terminate and return answer
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, find in Search BonBon tool first, then should be one of [{tool_names}]. If not, terminate answer
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat at most 5 times.)
Thought: I now know the final answer. Let's return the answer
Final Answer: the final answer to the original input question. If use Search BonBon then include the page of the PDF file that has the question

Begin!
Context:
{entities}

Current conversation:
Chat History:
{history}
Last line:
Human: {input}
Thought: {agent_scratchpad}
"""
prompt_react = hub.pull("hwchase17/react")
prompt_react.template = template
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_react)
react_agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=ConversationEntityMemory(llm=llm, top_k=3),
    max_iterations=3
)
i = input("enter promt ('exit' to terminate): ")
while i.lower() != "exit":
    query = i
    print("Responding ...")
    response = react_agent_executor.invoke({"input": query})
    result = response.get("output")
    print(result)
    i = input("enter new promt ('exit' to terminate): ")