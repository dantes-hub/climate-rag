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
st.set_page_config(page_title="RAG Climate QA", layout="centered")
st.title(" RAG Q&A Climate(New York) by Anka")

# Dropdown to choose dataset
qa_type = st.selectbox(
    "Choose Dataset Type",
    ["Historical Climate", "Forecast Climate"],
    index=0,
)

# Mode selector
mode = st.radio("Choose Answering Mode", ["RAG (with retrieval)", "GPT-only (no retrieval)"])

# Load embeddings
@st.cache_resource
def load_vectorstore(index_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Prompt
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

# Question box
question = st.text_input("Ask your climate question:")

# Answering logic
if question:
    with st.spinner("Thinking..."):

        # Choose vectorstore path based on dropdown
        if qa_type == "Historical Climate":
            index_path = "faiss_index"
        else:
            index_path = "faiss_index_forecast"

        if mode == "RAG (with retrieval)":
            vectorstore = load_vectorstore(index_path)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt}
            )
            response = qa_chain.invoke(question)
            st.subheader("Answer (RAG)")
            st.write(response["result"])

            st.subheader("Retrieved Sources")
            for doc in response["source_documents"]:
                st.markdown(f"- {doc.page_content}")
        else:
            response = llm.invoke(question)
            st.subheader("Answer (GPT Only)")
            st.write(response.content)
