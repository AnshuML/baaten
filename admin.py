import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_parser import get_pdf_text, get_text_chunks, save_vector_store

load_dotenv()

st.set_page_config(page_title="Admin Panel - Upload HR Docs")
st.title("📂 Admin: Upload & Index PDFs")

uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button("Submit & Index"):
    if uploaded_pdfs:
        with st.spinner("Processing and indexing..."):
            text = get_pdf_text(uploaded_pdfs)
            chunks = get_text_chunks(text)
            save_vector_store(chunks)
            st.success("✅ Indexed successfully!")
    else:
        st.warning("Please upload at least one PDF.")
