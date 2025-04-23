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

#   Dropbox FAISS index downloader
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

#   Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#   Page settings
st.set_page_config(page_title="RAG Assistant | Climate & Finance", layout="centered")
st.title("RAG Q&A Assistant by Anka")
st.markdown("Ask intelligent questions based on retrieved data from vector databases:")

#   Vector DB selector
qa_type = st.selectbox(
    "Select Vector Database Source",
    ["Historical Climate", "Forecast Climate", "Stock History"],
    index=0,
)

#   Sample questions
sample_qs = {
    "Stock History": [
        "What was the closing price of Tesla on July 2, 2020?",
        "What was Apple’s highest price in 2021?",
        "How did Microsoft perform during 2022?",
    ],
    "Historical Climate": [
        "What was the average temperature in New York in July 1990?",
        "How did the climate change between 1900 and 2000 in New York?",
    ],
    "Forecast Climate": [
        "What is the predicted temperature for New York in 2025?",
        "Is New York expected to warm in the next 5 years?",
    ]
}

if qa_type in sample_qs:
    selected_q = st.selectbox("Sample Questions", [""] + sample_qs[qa_type])
    if selected_q and st.button("Use Sample Question"):
        st.session_state["question"] = selected_q

# User input
question = st.text_input("Ask your question:", value=st.session_state.get("question", ""))

# Mode selector
mode = st.radio("Answering Mode", ["RAG (with retrieval)", "GPT-only (no retrieval)"])

#   Load FAISS with caching
@st.cache_resource
def load_vectorstore(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

#   Prompts
climate_prompt = PromptTemplate(
    template="""
You are a helpful climate assistant. Use the context below to answer the question.
Context:
{context}
Question: {question}
Answer:
""",
    input_variables=["context", "question"],
)

stock_prompt = PromptTemplate(
    template="""
You are a financial assistant helping users understand historical stock trends.
Always use precise numbers and context from the retrieved documents. Don't guess—only answer based on the provided context.
Context:
{context}
Question: {question}
Answer:
""",
    input_variables=["context", "question"],
)

#   Run QA
if question:
    with st.spinner("Thinking..."):

        # Select index path and prompt
        if qa_type == "Historical Climate":
            index_path = "faiss_index"
            prompt = climate_prompt
        elif qa_type == "Forecast Climate":
            index_path = "faiss_index_forecast"
            prompt = climate_prompt
        elif qa_type == "Stock History":
            index_path = "faiss_index/stock_index"
            prompt = stock_prompt
            download_and_extract_faiss_index()
        else:
            st.error("Unknown dataset type.")
            st.stop()

        # Load model + vectorstore
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        if mode == "RAG (with retrieval)":
            vectorstore = load_vectorstore(index_path)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            response = qa_chain.invoke(question)
            st.subheader("Answer (RAG)")
            st.write(response["result"])

            # Show retrieved text in code block style
            with st.expander("Retrieved Source Chunks"):
                for doc in response["source_documents"]:
                    st.code(doc.page_content)

            # Optional: chart for Stock History if date range detected
            if qa_type == "Stock History":
                date_range = re.findall(r"(20\d{2})", question)
                if len(date_range) >= 1:
                    ticker = None
                    for t in ["AAPL", "TSLA", "MSFT", "JNJ", "AMZN"]:
                        if t.lower() in question.lower():
                            ticker = t
                            break
                    if ticker:
                        try:
                            df = pd.read_csv(f"data/full_history/{ticker}.csv")
                            df["date"] = pd.to_datetime(df["date"], errors="coerce")
                            df = df.dropna(subset=["date"])
                            df = df.sort_values("date")
                            if len(date_range) == 1:
                                year = int(date_range[0])
                                df = df[df["date"].dt.year == year]
                            elif len(date_range) >= 2:
                                y1, y2 = int(date_range[0]), int(date_range[1])
                                df = df[(df["date"].dt.year >= y1) & (df["date"].dt.year <= y2)]
                            st.line_chart(df.set_index("date")["close"])
                        except Exception as e:
                            st.info(f"Chart error: {str(e)}")

        else:
            response = llm.invoke(question)
            st.subheader("Answer (GPT Only)")
            st.write(response.content)
