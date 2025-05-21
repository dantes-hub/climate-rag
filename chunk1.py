import pandas as pd
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

df = pd.read_csv("data/global_temperatures.csv")
df['dt'] = pd.to_datetime(df['dt'])
df = df.dropna(subset=["AverageTemperature"])
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


ny_df = df[df['City'] == "New York"].copy()
ny_df['year'] = ny_df['dt'].dt.year

# Chunk by year
year_docs = []
for year, group in ny_df.groupby('year'):
    temp_lines = [
        f"{row['dt'].strftime('%Y-%m-%d')}: {row['AverageTemperature']:.2f}°C"
        for _, row in group.iterrows()
    ]
    content = f"Temperature records for New York in {year}:\n" + "\n".join(temp_lines)
    year_docs.append(Document(page_content=content, metadata={"year": year}))

# Show some sample output
for doc in year_docs[:3]:
    print("", doc.metadata)
    print(doc.page_content[:300] + "\n...\n")

# embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Build and save FAISS index
vectorstore = FAISS.from_documents(year_docs, embedding_model)
vectorstore.save_local("faiss_index")
