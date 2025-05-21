import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load preprocessed chunks
with open("stock_chunks.pkl", "rb") as f:
    chunk_data = pickle.load(f)

# Build LangChain documents
docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunk_data]

# Create embedding model
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Build FAISS index and save it
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("faiss_index/stock_index")

print("Vector store created and saved.")
