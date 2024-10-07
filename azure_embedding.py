import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(verbose=True, override=True)

file_path = ".\data\BonBon FAQ.pdf"
loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(document)

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, "chroma_data")

chroma_client = chromadb.PersistentClient(
    path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False)
)

openai_ef = OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_type="azure",
    api_version="2024-06-01",
    model_name="text-embedding-ada-002",
)

collection = chroma_client.get_or_create_collection(
    name=os.getenv("CHROMA_COLLECTION_NAME"), embedding_function=openai_ef
)

collection.add(
    ids=[str(i) for i, _ in enumerate(chunked_documents)],
    documents=[document.page_content for document in chunked_documents],
)

results = collection.query(
    query_texts=["What does NashTech Business Process Outsourcing Team does?"],
    n_results=1,
)
print(results)
