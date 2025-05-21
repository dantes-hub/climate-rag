import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests, zipfile, io
import re
from datetime import datetime

# Dropbox FAISS index downloader
def download_and_extract_faiss_index():
    index_dir = "faiss_index/stock_index"
    if not os.path.exists(f"{index_dir}/index.faiss"):
        os.makedirs(index_dir, exist_ok=True)
        st.info("Downloading FAISS index from Dropbox...")
        url = "https://www.dropbox.com/scl/fi/bajzvodai1zvwf6h7n6vt/index.zip?rlkey=apuxm3tiuuy5jielyne1xb0mv&st=f74v78kb&dl=1"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(index_dir)
        st.success("FAISS index downloaded and extracted.")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page settings
st.set_page_config(page_title="RAG Assistant | Climate & Finance", layout="centered")
st.title("RAG Q&A Assistant by Ankha")
st.markdown("Ask intelligent questions based on retrieved data from AI (RAG):")

# Layout split: left = controls, right = question/results
col1, col2 = st.columns([1, 2])

with col1:
    qa_type = st.selectbox(
        "Datasets & Action",
        ["Historical Climate", "Forecast Climate", "Stock History", "Stock Forecast"],
        index=0,
    )

    mode = st.radio("Answering Mode", ["RAG (with retrieval)", "GPT-only (no retrieval)"])

    sample_qs = {
        "Stock History": [
            "Show a chart of Tesla stock in 2017",
            "What was Apple’s highest price in 2010?",
            "Plot Microsoft trend during 2019",
        ],
        "Historical Climate": [
            "What was the average temperature in New York in July 1990?",
            "How did the climate change between 1900 and 2000 in New York?",
        ],
        "Forecast Climate": [
            "How will the climate in New York be in 2026?",
            "Is it getting hotter in New York between 2025 and 2028?",
        ],
        "Stock Forecast": [
            "What is the predicted price of Apple in 2028?",
            "Show Tesla stock forecast in 2026",
            "How will Microsoft stock trend in 2027?",
        ]

    }

    selected_q = st.selectbox("Sample Questions", [""] + sample_qs.get(qa_type, []))
    if selected_q:
        st.session_state["question"] = selected_q

with col2:
    question = st.text_input("Ask your question:", value=st.session_state.get("question", ""))

@st.cache_resource
def load_vectorstore(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

climate_prompt = PromptTemplate(
    template="""You are a helpful climate assistant. Use the context below to answer the question.
Context:
{context}
Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

stock_prompt = PromptTemplate(
    template="""You are a financial assistant helping users understand historical stock trends.
Always use precise numbers and context from the retrieved documents. Don't guess—only answer based on the provided context.
Context:
{context}
Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

if question:
    with st.spinner("Thinking..."):

        if qa_type == "Historical Climate":
            index_path = "faiss_index"
            prompt = climate_prompt
        elif qa_type == "Forecast Climate":
            index_path = "faiss_index_forecast"
            prompt = climate_prompt
        elif qa_type == "Stock Forecast":
            index_path = "faiss_index/stock_forecast_2026_2030"
            prompt = stock_prompt    
        elif qa_type == "Stock History":
            index_path = "faiss_index/stock_index"
            prompt = stock_prompt
            download_and_extract_faiss_index()
        else:
            st.error("Unknown dataset type.")
            st.stop()

        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

        if mode == "RAG (with retrieval)":
            vectorstore = load_vectorstore(index_path)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            response = qa_chain.invoke(question)

            st.markdown("### Answer")
            st.markdown(f"<div style='font-size: 16px; line-height: 1.6;'>{response['result']}</div>", unsafe_allow_html=True)

            with st.expander("Retrieved Source Chunks"):
                for doc in response["source_documents"]:
                    st.code(doc.page_content)

            # Chart logic: trigger only if user mentions "chart", etc.
            if qa_type == "Stock History":
                question_lower = question.lower()
                chart_keywords = ["chart", "plot", "graph", "trend", "visual"]
                date_range = re.findall(r"(20\d{2})", question_lower)

                company_to_ticker = {
                    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
                    "amazon": "AMZN", "johnson": "JNJ"
                }

                ticker = None
                for name, symbol in company_to_ticker.items():
                    if name in question_lower:
                        ticker = symbol
                        break

                if ticker and date_range and any(k in question_lower for k in chart_keywords):
                    try:
                        df = pd.read_csv(f"data/full_history/{ticker}.csv")
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"]).sort_values("date")

                        if len(date_range) == 1:
                            year = int(date_range[0])
                            df = df[df["date"].dt.year == year]
                        elif len(date_range) >= 2:
                            y1, y2 = int(date_range[0]), int(date_range[1])
                            df = df[(df["date"].dt.year >= y1) & (df["date"].dt.year <= y2)]

                        if not df.empty:
                            st.markdown(f"#### {ticker} Price Trend")
                            st.line_chart(df.set_index("date")["close"])
                    except Exception as e:
                        st.error(f"Chart error: {str(e)}")

        else:
            response = llm.invoke(question)
            st.markdown("### Answer")
            st.markdown(f"<div style='font-size: 16px; line-height: 1.6;'>{response.content}</div>", unsafe_allow_html=True)
