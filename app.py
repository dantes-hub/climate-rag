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

# UI config
st.set_page_config(page_title="RAG Climate QA", layout="centered")
st.title("RAG Q&A on New York Climate by Anka")
st.markdown("Ask a question like: *What was the temperature in New York in 1750?*")

# Mode selector
mode = st.radio(
    "Choose QA Mode:",
    ["RAG (with retrieval)", "GPT-only (no retrieval)"],
    index=0,
    help="RAG uses vector DB, GPT-only uses no context"
)

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# RAG prompt
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

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# User input
user_question = st.text_input("Your question:", placeholder="e.g., What was the average temp in 1800?")
if user_question:
    with st.spinner("Thinking..."):
        if mode == "RAG (with retrieval)":
            response = qa_chain.invoke(user_question)
            st.subheader("Answer (RAG)")
            st.write(response["result"])

            st.subheader("Retrieved Sources")
            for doc in response["source_documents"]:
                st.markdown(f"- {doc.page_content}")

        else:
            response = llm.invoke(user_question)
            st.subheader("Answer (GPT-Only)")
            st.write(response.content)
