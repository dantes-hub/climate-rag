import os
import pandas as pd

def generate_stock_chunks(data_path='data/full_history'):
    chunks = []

    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            ticker = filename.replace('.csv', '')
            file_path = os.path.join(data_path, filename)
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                try:
                    date = row['date']
                    open_p = float(row['open'])
                    high = float(row['high'])
                    low = float(row['low'])
                    close = float(row['close'])
                    volume = int(row['volume'])

                    text = (
                        f"On {date}, {ticker} opened at ${open_p:.2f}, reached a high of ${high:.2f}, "
                        f"a low of ${low:.2f}, and closed at ${close:.2f}. Volume traded was {volume:,} shares."
                    )

                    metadata = {
                        "ticker": ticker,
                        "date": date,
                        "type": "stock_history"
                    }

                    chunks.append({"text": text, "metadata": metadata})

                except Exception as e:
                    print(f"Error in {ticker} on {date}: {e}")

    return chunks

# Save as .pkl or return for embedding
if __name__ == "__main__":
    import pickle
    chunks = generate_stock_chunks()
    with open("stock_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
