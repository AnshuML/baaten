# admin.py - Secure Admin Panel with Persistent Upload, Process & Delete

import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from utils.pdf_parser import get_pdf_text, get_text_chunks, save_vector_store

# Load env variables
load_dotenv()

# Constants
UPLOAD_DIR = "hr/uploads"
PROCESSED_DIR = "hr/processed"
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------- Admin Login --------------------- #
def login():
    st.title("🔒 Admin Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if user == ADMIN_USER and password == ADMIN_PASS:
            st.session_state.logged_in = True
        else:
            st.error("❌ Invalid credentials")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --------------------- Admin Panel --------------------- #
st.set_page_config(page_title="Admin Panel")
st.title("📂 Admin Panel: Upload, Process, Delete")

# File Upload (Saved persistently)
uploaded_files = st.file_uploader("Upload HR PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

# List existing files
files = os.listdir(UPLOAD_DIR)
st.subheader("📑 Uploaded Files:")

if files:
    for file in files:
        col1, col2 = st.columns([6, 1])
        col1.write(f"📄 {file}")
        if col2.button("🗑️ Delete", key=file):
            confirm = st.warning(f"Are you sure to delete {file}?", icon="⚠️")
            if st.button("Yes, Delete", key=file+"_confirm"):
                os.remove(os.path.join(UPLOAD_DIR, file))
                st.experimental_rerun()
else:
    st.info("No files uploaded yet.")

# Submit and Process button
if st.button("🔄 Submit & Process", use_container_width=True):
    try:
        all_text = ""
        for file in os.listdir(UPLOAD_DIR):
            with open(os.path.join(UPLOAD_DIR, file), "rb") as f:
                all_text += get_pdf_text([f])

        chunks = get_text_chunks(all_text)
        save_vector_store(chunks)

        # Move processed files
        for file in os.listdir(UPLOAD_DIR):
            shutil.move(os.path.join(UPLOAD_DIR, file), os.path.join(PROCESSED_DIR, file))

        st.success("✅ Documents processed and indexed successfully!")
    except Exception as e:
        st.error(f"❌ Error: {e}")

