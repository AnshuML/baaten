import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # ✅ Correct FAISS Import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai


# ✅ Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Check and install FAISS
try:
    import faiss
    print("FAISS is installed successfully!")
except ImportError:
    st.error("FAISS is not installed! Please install it using `pip install faiss-cpu` or `pip install faiss-gpu` if you have a CUDA-supported GPU.")
    st.stop()

# ✅ Function to extract text from multiple PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ✅ Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# ✅ Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")  # ✅ Save FAISS Index
        print("FAISS vector store created successfully!")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")

# ✅ Function to create conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say "The answer is not available in the context". Don't guess or provide incorrect answers.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ✅ Function to handle user input and search FAISS index
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        
        if not docs:
            st.warning("No relevant information found in the uploaded PDFs.")
            return
        
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.write("**Reply:**", response["output_text"])
    
    except Exception as e:
        st.error(f"Error during question answering: {e}")

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("📄 Hr Mind")
    
    with st.sidebar:
        st.title("📌 Upload PDFs:")
        pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("🔄 Processing PDFs..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Processing Done Successfully!")
                else:
                    st.error("⚠️ Please upload at least one PDF file.")

    user_question = st.text_input("💬 Ask a question from the uploaded PDFs:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
