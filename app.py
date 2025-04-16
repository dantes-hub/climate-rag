import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables (only used locally)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# UI config
st.set_page_config(page_title="RAG Climate QA", layout="centered")
st.title(" Retrieval-Augmented Q&A on New York Climate ,Ankama")
st.markdown("Ask a question like: *What was the temperature in New York in 1750?*")

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Build RAG chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# User input
user_question = st.text_input("Your question:", placeholder="e.g., What was the average temp in 1800?")
if user_question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(user_question)
        st.subheader("Answer")
        st.write(response["result"])
        st.subheader("Retrieved Sources")
        for doc in response["source_documents"]:
            st.markdown(f"- {doc.page_content}")
