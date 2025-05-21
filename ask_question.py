import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the saved FAISS index
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Set up the LLM (you can switch to gpt-4 if you want)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Custom prompt for better reasoning with numeric data
custom_prompt = PromptTemplate(
    template="""
You are a helpful weather assistant. Use the context below to answer the question.
If the context contains multiple monthly temperatures from the same year, estimate the yearly average temperature based on available data.
Always provide a clear answer using the retrieved values.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"],
)

# Create the RAG chain using the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Interactive Q&A loop
while True:
    question = input("\n🤔 Ask a question (or type 'exit'): ")
    if question.lower() == "exit":
        break

    response = qa_chain.invoke(question)

    print("\n🧠 Answer:")
    print(response["result"])

    print("\n📚 Top source chunks:")
    for doc in response["source_documents"][:2]:
        print(f"- {doc.page_content}")
