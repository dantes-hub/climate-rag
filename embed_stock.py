# embed_stock.py
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def build_stock_index():
    docs = []

    for file in os.listdir("data/full_history"):
        if file.endswith(".csv"):
            ticker = file.replace('.csv', '')
            path = os.path.join("data/full_history", file)
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                try:
                    text = (
                        f"On {row['date']}, {ticker} opened at ${float(row['open']):.2f}, "
                        f"reached a high of ${float(row['high']):.2f}, a low of ${float(row['low']):.2f}, "
                        f"and closed at ${float(row['close']):.2f}. Volume traded was {int(row['volume']):,} shares."
                    )
                    metadata = {
                        "ticker": ticker,
                        "date": row['date'],
                        "type": "stock_history"
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
                except:
                    continue

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index/stock_index")
