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

# Page settings
st.set_page_config(page_title="RAG Assistant | Climate & Finance", layout="centered")
st.title("RAG Q&A Assistant by Anka")

st.markdown("Ask intelligent questions about historical **climate** in New York or explore **stock market performance** over the years.")

# Dataset selector
qa_type = st.selectbox(
    "🔍 Choose Dataset Domain",
    ["Historical Climate", "Forecast Climate", "Stock History"],
    index=0,
)

# Mode selector
mode = st.radio("Answering Mode", ["RAG (with retrieval)", "GPT-only (no retrieval)"])

# Embedding cache
@st.cache_resource
def load_vectorstore(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Custom prompts per domain
climate_prompt = PromptTemplate(
    template="""
You are a helpful climate assistant. Use the context below to answer the question.
If the context contains multiple monthly temperatures from the same year, estimate the yearly average temperature based on available data.

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

# Question input
question = st.text_input("❓ Ask your question:")

# Logic
if question:
    with st.spinner("Thinking..."):

        # Dataset-specific configuration
        if qa_type == "Historical Climate":
            index_path = "faiss_index"
            prompt = climate_prompt
        elif qa_type == "Forecast Climate":
            index_path = "faiss_index_forecast"
            prompt = climate_prompt
        elif qa_type == "Stock History":
            index_path = "faiss_index/stock_index"
            prompt = stock_prompt
            
            if not os.path.exists("faiss_index/stock_index/index.faiss"):
                with st.spinner("Rebuilding FAISS index for Stock History..."):
                    from embed_stock import build_stock_index
                    build_stock_index()

        else:
            st.error("Unknown dataset type.")
            st.stop()

        # LLM setup
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

            with st.expander("Retrieved Source Chunks"):
                for doc in response["source_documents"]:
                    st.markdown(f"- {doc.page_content}")

        else:
            response = llm.invoke(question)
            st.subheader("Answer (GPT Only)")
            st.write(response.content)
