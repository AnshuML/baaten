import streamlit as st
import os
from dotenv import load_dotenv
from utils.qa_chain import get_conversational_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(page_title="HR Mind")
st.header("🧠 Hr Mind - Ask HR Docs")

question = st.text_input("💬 Ask a question:")

if question:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(question)

        if not docs:
            st.warning("No relevant info found.")
        else:
            chain = get_conversational_chain()
            response = chain.invoke({"input_documents": docs, "question": question})
            st.markdown("**Reply:** " + response["output_text"])

    except Exception as e:
        st.error(f"❌ Error: {e}")
