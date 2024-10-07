import os

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)
# chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")), settings=Settings())
DIR = os.path.dirname(os.path.abspath(__file__))
print(DIR)
DB_PATH = os.path.join(DIR, "chroma_data")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False)
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # TODO: put the path to the PDF file to index here
file_path = ".\data\BonBon FAQ.pdf"

loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(document)

Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function,
    collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    client=chroma_client,
)
print(f"Added {len(chunked_documents)} chunks to chroma db")

collection = chroma_client.get_collection(name=os.getenv("CHROMA_COLLECTION_NAME"))

results = collection.query(
    query_texts=["What does NashTech Business Process Outsourcing Team does?"],
    n_results=1,
)

print(results)


