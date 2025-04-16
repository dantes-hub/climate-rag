# baseline_gpt_only.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

while True:
    question = input("\n🤔 Ask a question (baseline, no RAG): ")
    if question.lower() == "exit":
        break

    response = llm.invoke(question)
    print("\n🧠 GPT-only Answer:")
    print(response.content)
