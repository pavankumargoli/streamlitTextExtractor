import streamlit as st
import PyPDF2
import pdfplumber
import os
import io
import sqlite3
from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

def get_vectors_from_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, file_type, content, embedding FROM texts")
    records = cursor.fetchall()

    vectors = {}
    for record in records:
        file_name, file_type, content, embedding_bytes = record
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)  # Convert bytes to numpy array
        vectors[file_name] = {'file_type': file_type, 'content': content, 'embedding': embedding}

    return vectors

# Functions to extract text
def extract_text_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# Functions for database operations
def create_database():
    conn = sqlite3.connect('file_texts.db')
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS texts") 
    cursor.execute('''CREATE TABLE IF NOT EXISTS texts
                      (file_name TEXT PRIMARY KEY, file_type TEXT, content TEXT, embedding BLOB)''')
    conn.commit()
    return conn

def store_text_in_db(conn, file_name, file_type, content, embedding):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO texts (file_name, file_type, content, embedding) VALUES (?, ?, ?, ?)",
                   (file_name, file_type, content, embedding))
    conn.commit()

# Function to generate embeddings
def generate_embedding(text, model):
    embedding = model.encode(text)
    return sqlite3.Binary(embedding.tobytes())

# Main Streamlit application
def main():
    st.set_page_config(page_title="Text Extractor", page_icon=None, layout="centered", initial_sidebar_state="auto")

    st.title("Text Extractor")
    st.subheader("Extract text from PDF and DOCX files and store it in a database")

    file = st.file_uploader("Choose a file (PDF or DOCX)", type=['pdf', 'docx'])

    # Initialize database and model
    conn = create_database()
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    if file is not None:
        file_name = file.name
        file_type = ''

        if file.type == 'application/pdf':
            with st.spinner('Extracting text from PDF...'):
#                 display_pdf(file)
                file.seek(0)
                text = extract_text_pdf(file)
                file_type = 'PDF'
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            with st.spinner('Extracting text from DOCX...'):
                text = extract_text_docx(file)
                file_type = 'DOCX'
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX file.")

        if text:
            # Generate embedding and store text and embedding in the database
            embedding = generate_embedding(text, model)
            store_text_in_db(conn, file_name, file_type, text, embedding)
            st.success(f"Successfully stored {file_name} in the database.")

            st.subheader("Extracted Text")
            st.write(text)
        else:
            st.error("Unable to process the file. No text was extracted.")

    
    st.subheader("Saved Text Records and Vectors in Database")
    records = get_vectors_from_db(conn)
    for file_name, data in records.items():
        st.write(f"File: {file_name} ({data['file_type']})")
        st.write(f"Content (first 100 characters): {data['content'][:100]}...")
        st.write(f"Vector (first 5 elements): {data['embedding'][:5]}")
        st.write("----")
    
    conn.close()

if __name__ == '__main__':
    main()
