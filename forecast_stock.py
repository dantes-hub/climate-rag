import pandas as pd
from prophet import Prophet
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Stock files
companies = {
    "AAPL": "data/full_history/AAPL.csv",
    "TSLA": "data/full_history/TSLA.csv",
    "MSFT": "data/full_history/MSFT.csv",
    "AMZN": "data/full_history/AMZN.csv",
    "JNJ":  "data/full_history/JNJ.csv"
}

forecast_docs = []

for ticker, path in companies.items():
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'close'])

    prophet_df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df = prophet_df.dropna()

    # Train model
    model = Prophet(daily_seasonality=False)
    model.fit(prophet_df)

    # Generate enough future periods to reach 2030
    future = model.make_future_dataframe(periods=365 * 15)
    forecast = model.predict(future)

    # Keep only 2026–2030
    forecast_future = forecast[(forecast['ds'].dt.year >= 2026) & (forecast['ds'].dt.year <= 2030)][['ds', 'yhat']]
    forecast_future['year'] = forecast_future['ds'].dt.year

    for year, group in forecast_future.groupby('year'):
        lines = [
            f"{row['ds'].strftime('%Y-%m-%d')}: ${row['yhat']:.2f}"
            for _, row in group.iterrows()
        ]
        content = f"Forecasted closing prices for {ticker} in {year}:\n" + "\n".join(lines)
        forecast_docs.append(Document(page_content=content, metadata={"ticker": ticker, "year": year}))

print(f"Prepared {len(forecast_docs)} forecast documents.")

# Embed & save
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(forecast_docs, embedding_model)
vectorstore.save_local("faiss_index/stock_forecast_2026_2030")

print("Saved forecast vectorstore to 'faiss_index/stock_forecast_2026_2030'")
