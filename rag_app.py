import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  



# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
df = pd.read_csv("data/global_temperatures.csv")
df = df.dropna(subset=["AverageTemperature"])

# Filter for New York or keep all
df = df[df["City"] == "New York"].reset_index(drop=True)

# Create simple text format per row
docs = []
for _, row in df.iterrows():
    text = f"On {row['dt']}, in {row['City']}, the average temperature was {row['AverageTemperature']:.2f}°C."
    metadata = {"city": row["City"], "date": row["dt"]}
    docs.append(Document(page_content=text, metadata=metadata))

print(f" Prepared {len(docs)} text documents.")

# Embed and store in FAISS
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save FAISS index
vectorstore.save_local("faiss_index")

print(" Vector store created and saved as 'faiss_index'")
