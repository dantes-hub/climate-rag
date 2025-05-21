import pandas as pd
import numpy as np
from prophet import Prophet
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
df = pd.read_csv("data/global_temperatures.csv")
df['dt'] = pd.to_datetime(df['dt'])
df = df.dropna(subset=['AverageTemperature'])

# Parse New York latitude
ny_lat = df[df['City'] == 'New York'].iloc[0]['Latitude']
ny_lat_value = float(ny_lat[:-1]) * (1 if ny_lat.endswith('N') else -1)

def parse_lat(lat_str):
    return float(lat_str[:-1]) * (1 if lat_str.endswith('N') else -1)

df['ParsedLatitude'] = df['Latitude'].apply(parse_lat)

# Filter ±1° neighbors
lat_range = 1.0
df_neighbors = df[np.abs(df['ParsedLatitude'] - ny_lat_value) <= lat_range]

# NY data
ny_data = df_neighbors[df_neighbors['City'] == 'New York'][['dt', 'AverageTemperature']]
ny_data = ny_data.rename(columns={'AverageTemperature': 'NY_Temp'})

# Neighbor avg
neighbor_data = df_neighbors[df_neighbors['City'] != 'New York']
neighbor_avg = neighbor_data.groupby('dt')['AverageTemperature'].mean().reset_index()
neighbor_avg = neighbor_avg.rename(columns={'AverageTemperature': 'Neighbor_AvgTemp'})

# Merge
merged = pd.merge(ny_data, neighbor_avg, on='dt')
merged = merged.rename(columns={'dt': 'ds', 'NY_Temp': 'y'})
merged = merged.dropna()
merged['ds'] = pd.to_datetime(merged['ds'])
merged['y'] = pd.to_numeric(merged['y'])
merged['Neighbor_AvgTemp'] = pd.to_numeric(merged['Neighbor_AvgTemp'])

# Filter from 1850 to avoid overflow and use modern trends
merged = merged[merged['ds'].dt.year >= 1850]

# Train model
model = Prophet()
model.add_regressor('Neighbor_AvgTemp')
model.fit(merged[['ds', 'y', 'Neighbor_AvgTemp']])

# Forecast next 30 years
future = model.make_future_dataframe(periods=12 * 30, freq='MS')
neighbor_avg_extended = merged['Neighbor_AvgTemp'].tolist()
future_extension = [merged['Neighbor_AvgTemp'].mean()] * (len(future) - len(merged))
future['Neighbor_AvgTemp'] = neighbor_avg_extended + future_extension

# Predict
forecast = model.predict(future)

# Create forecast doc chunks (one per year)
last_year = merged['ds'].dt.year.max()
forecast_future = forecast[forecast['ds'].dt.year > last_year][['ds', 'yhat']]
forecast_future['year'] = forecast_future['ds'].dt.year

forecast_docs = []
for year, group in forecast_future.groupby('year'):
    lines = [
        f"{row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.2f}°C"
        for _, row in group.iterrows()
    ]
    content = f"Forecasted temperatures for New York in {year}:\n" + "\n".join(lines)
    forecast_docs.append(Document(page_content=content, metadata={"year": year}))

print(f" Prepared {len(forecast_docs)} forecast documents.")

# Embed & save to FAISS
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(forecast_docs, embedding_model)
vectorstore.save_local("faiss_index_forecast")

print(" Saved forecast vectorstore to 'faiss_index_forecast'")
