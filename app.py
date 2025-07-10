import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import langdetect
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.retrievers import BM25Retriever
import pytz
from googletrans import Translator
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Greeting logic
ist = pytz.timezone('Asia/Kolkata')
hour = datetime.now(ist).hour
if 5 <= hour < 12:
    greeting = "Good morning!"
elif 12 <= hour < 18:
    greeting = "Good afternoon!"
elif 18 <= hour < 22:
    greeting = "Good evening!"
else:
    greeting = "Good night!"

# Set Streamlit page config
st.set_page_config(page_title="HR Mind", layout="centered", page_icon="💼")

# Stylish header
st.markdown(f"""
<div style='
    background-color:#102027;
    border-radius:18px;
    padding:30px;
    text-align:center;
    margin: 20px auto;
    max-width:600px;
'>
    <h1 style='color:white;'>HR Mind</h1>
    <div style='color:#f50057; font-weight:700;'>AIPL Group</div>
    <div style='color:#ffeb3b; font-weight:600; margin-top:10px;'>{greeting}</div>
</div>
""", unsafe_allow_html=True)

# Language mapping
language_options = {
    'English': 'en',
    'Hindi (हिन्दी)': 'hi',
    'Tamil (தமிழ்)': 'ta',
    'Telugu (తెలుగు)': 'te',
    'Bengali (বাংলা)': 'bn',
    'Malayalam (മലയാളം)': 'ml',
    'Kannada (ಕನ್ನಡ)': 'kn',
    'Gujarati (ગુજરાતી)': 'gu',
    'Marathi (मराठी)': 'mr',
    'Urdu (اردو)': 'ur',
    'Odia (ଓଡ଼ିଆ)': 'or',
    'Assamese (অসমীয়া)': 'as',
    'Punjabi (ਪੰਜਾਬੀ)': 'pa',
    'Sindhi (سنڌي)': 'sd',
    'Yoruba (Nigeria)': 'yo',
    'Igbo (Nigeria)': 'ig',
    'Hausa (Nigeria)': 'ha',
    'Sinhala (සිංහල)': 'si',
}

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Reranking using BM25
def bm25_rerank(question, docs, top_n=5):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    reranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked_docs[:top_n]

# Load embeddings and vectorstore once
@st.cache_resource
def load_resources():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return embeddings, db

embeddings, db = load_resources()

# Language selection dropdown
selected_language = st.selectbox("🌐 Select response language:", list(language_options.keys()))
selected_lang_code = language_options[selected_language]

# User input
question = st.text_input("", placeholder="💬 Ask a question from HR Constitution...", key="user_input")

if question:
    try:
        # Remove any language tag inside parentheses like (Hindi) or [Tamil] from question
        clean_question = re.sub(r"$[^)]*$|\[[^\]]*\]$", "", question).strip()

        # Retrieve top 15 documents
        docs = db.similarity_search(clean_question, k=15)
        if not docs:
            st.warning("⚠️ No relevant information found.")
        else:
            # Rerank top 15 docs using BM25
            reranked_docs = bm25_rerank(clean_question, docs, top_n=5)

            # Create BM25 Retriever
            tokenizer = lambda text: text.split()
            retriever = BM25Retriever.from_documents(reranked_docs, tokenizer=tokenizer)

            # Build prompt template
            prompt_template = """
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:
            """
            PROMPT = PromptTemplate.from_template(prompt_template)

            # Initialize Gemini LLM
            llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

            # Build QA chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="question",
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            # Get response
            response = chain.invoke({"question": clean_question})
            result_str = str(response.get("result", ""))

            # Translate if needed
            if selected_lang_code != 'en':
                translator = Translator()
                translated = translator.translate(result_str, dest=selected_lang_code)
                result_str = translated.text

            # Display chat messages
            st.chat_message("user").markdown(str(question))
            st.chat_message("assistant").markdown(result_str)

            # Save to chat history
            st.session_state.chat_history.append({"role": "user", "content": str(question)})
            st.session_state.chat_history.append({"role": "assistant", "content": result_str})

    except Exception as e:
        st.error(f"❌ Error: {e}")