import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# UI setup
st.set_page_config(page_title="RAG Climate QA", layout="wide")
st.title("🌎 RAG Q&A on New York Climate by Anka")

# Load embeddings
@st.cache_resource
def load_vectorstore(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Shared prompt
custom_prompt = PromptTemplate(
    template="""
You are a helpful weather assistant. Use the context below to answer the question.
If the context contains multiple monthly temperatures from the same year, estimate the yearly average temperature based on available data.
Be precise and reference actual values when possible.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"],
)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Tabs
tab1, tab2 = st.tabs(["📜 Historical QA", "🔮 Forecast QA"])

# ---------------- Tab 1: Historical ----------------
with tab1:
    st.header("Historical Climate Questions")
    mode_hist = st.radio("Choose Mode:", ["RAG", "GPT-Only"], key="hist_mode")
    user_question_hist = st.text_input("Ask something (e.g., What was the temp in 1800?)", key="hist_q")

    if user_question_hist:
        with st.spinner("Answering..."):
            if mode_hist == "RAG":
                vectorstore = load_vectorstore("faiss_index")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": custom_prompt}
                )
                response = qa_chain.invoke(user_question_hist)
                st.subheader("Answer (RAG)")
                st.write(response["result"])
                st.subheader("Retrieved Sources")
                for doc in response["source_documents"]:
                    st.markdown(f"- {doc.page_content}")
            else:
                response = llm.invoke(user_question_hist)
                st.subheader("Answer (GPT Only)")
                st.write(response.content)

# ---------------- Tab 2: Forecast ----------------
with tab2:
    st.header("Forecast Climate Questions")
    mode_fore = st.radio("Choose Mode:", ["RAG", "GPT-Only"], key="forecast_mode")
    user_question_forecast = st.text_input("Ask something (e.g., Predicted temp in 2035?)", key="forecast_q")

    if user_question_forecast:
        with st.spinner("Answering..."):
            if mode_fore == "RAG":
                vectorstore = load_vectorstore("faiss_index_forecast")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": custom_prompt}
                )
                response = qa_chain.invoke(user_question_forecast)
                st.subheader("Answer (RAG)")
                st.write(response["result"])
                st.subheader("Retrieved Forecast Sources")
                for doc in response["source_documents"]:
                    st.markdown(f"- {doc.page_content}")
            else:
                response = llm.invoke(user_question_forecast)
                st.subheader("Answer (GPT Only)")
                st.write(response.content)
