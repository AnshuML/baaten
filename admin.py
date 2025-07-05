import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Text extraction from PDFs
def get_pdf_text(files):
    text = ""
    for file in files:
        try:
            reader = PdfReader(file)
            file_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    file_text += extracted
            if not file_text.strip():
                st.warning(f"⚠️ No readable text found in file: {file.name}")
            else:
                text += file_text
        except Exception as e:
            st.warning(f"⚠️ Could not read file: {file.name} - {e}")
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return splitter.split_text(text)

# Streamlit UI
st.set_page_config(page_title="HR Admin Panel", layout="centered")

def admin_login():
    st.title("🔒 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

def admin_panel():
    st.title("📂 Admin Panel: Upload & Process HR Policies")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join("uploads", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f" Uploaded {len(uploaded_files)} file(s)")

    files = os.listdir("uploads") if os.path.exists("uploads") else []
    if files:
        st.subheader("📑 Current Files")
        for file in sorted(files):
            col1, col2 = st.columns([5, 1])
            col1.write(f"📄 {file}")
            if col2.button("🗑️", key=f"delete_{file}"):
                os.remove(os.path.join("uploads", file))
                st.success(f"Deleted {file}")
                st.rerun()
    else:
        st.info("No files uploaded yet.")

    if st.button("🔄 Process & Index", use_container_width=True):
        try:
            pdf_paths = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".pdf")]
            if not pdf_paths:
                st.warning("⚠️ No PDFs to process")
                return

            all_text = ""
            for path in pdf_paths:
                try:
                    with open(path, "rb") as f:
                        file_text = get_pdf_text([f])
                        if file_text.strip():
                            all_text += file_text
                        else:
                            st.warning(f"⚠️ Skipped file with no text: {path}")
                except Exception as e:
                    st.warning(f"⚠️ Error reading file {path}: {e}")

            if not all_text.strip():
                st.warning("⚠️ No valid text found in any uploaded PDFs. Cannot proceed.")
                return

            chunks = get_text_chunks(all_text)
            st.info(f"Split into {len(chunks)} chunks")

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            db = FAISS.from_texts(chunks, embedding=embeddings)
            db.save_local("faiss_index")
            st.success("✅ Processed and indexed successfully!")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    admin_login()
else:
    admin_panel()