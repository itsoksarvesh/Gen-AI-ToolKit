import os
import numpy as np
import faiss
import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

# Constants for optimization
MAX_DOC_LENGTH = 5000  # Max length of documents for LLM

# Streamlit title
st.title("🌍 Full-Content Translation with Hugging Face Embeddings and FAISS")

@st.cache_resource
def get_ollama_llm():
    return Ollama(model="llama3.2:3b")

ollama_llm = get_ollama_llm()

# Initialize FAISS index with the correct dimension for the selected model
@st.cache_resource
def initialize_faiss_index(dimension=384):  
    return faiss.IndexFlatL2(dimension)

# Function to get Hugging Face model
@st.cache_resource
def get_huggingface_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to compute embeddings for the text using Hugging Face
def get_text_embedding(text):
    model = get_huggingface_model()
    embedding = model.encode(text)
    return embedding.reshape(1, -1).astype(np.float32)  # Should be (1, 384)

# Truncate long text
def truncate_text(text, max_length=MAX_DOC_LENGTH):
    return text[:max_length] + ("..." if len(text) > max_length else "")

# Add embedding to FAISS index
def add_to_faiss_index(index, embedding):
    index.add(embedding)

# Pipeline for full content translation
def full_content_translation_pipeline(documents, target_language):
    combined_content = "\n\n".join(documents)  # Combine all document contents
    task_prompt = f"Translate the following content to {target_language}:\n\n{combined_content}"

    return analyze_with_ollama(task_prompt)

# Use Ollama for translation (using LLM)
def analyze_with_ollama(content):
    template = "{input_text}"
    prompt = PromptTemplate(input_variables=["input_text"], template=template)
    llm_chain = LLMChain(llm=ollama_llm, prompt=prompt)
    return llm_chain.run(content)

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# File uploader in Streamlit
uploaded_files = st.file_uploader("Upload PDF/Text files", accept_multiple_files=True, type=['pdf', 'txt'])

documents = []
faiss_index = initialize_faiss_index()

if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")

        text = truncate_text(text)  # Truncate large text
        documents.append(text)
        embedding = get_text_embedding(text)
        add_to_faiss_index(faiss_index, embedding)
    
    st.success(f"Indexed {len(documents)} documents!")
else:
    st.info("Please upload files.")

# Language selection
languages = ['Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Hindi', 'Arabic', 'Tamil', 'Bangla', 'Sanskrit']
target_language = st.selectbox("Select target language:", languages)

# Button for translation
if st.button("Translate Entire Content"):
    if documents:
        result = full_content_translation_pipeline(documents, target_language)
        st.subheader(f"Full Translation to {target_language}:")
        st.write(result)
    else:
        st.warning("Upload some documents.")
