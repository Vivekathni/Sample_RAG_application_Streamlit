# pages/Page1.py
import streamlit as st
import os
import psycopg2
import numpy as np
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

st.title("Upload a PDF File")
# Set up the title of the app
#st.title("PDF File Uploader")

# Create a file uploader widget for PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Add a button to show the file name when clicked
if uploaded_file is not None:
    # Save the uploaded file to the local directory
    save_folder = "./uploaded_files"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Create the full path to save the file
    save_path = os.path.join(save_folder, uploaded_file.name)
    
    # Save the file locally
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved successfully at: {save_path}")

    # Add a button to display the file name
    if st.button("Read Uploaded File"):
    
            
            pdf_folder_location = "uploaded_files"
            # loading pdf files from dataset folder
            pdf_loader = PyPDFDirectoryLoader(pdf_folder_location) # type: ignore
            # Create text_splitter
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                 encoding_name='cl100k_base',
                 chunk_size=512,
                 chunk_overlap=16
            )

            # Create chunks
            texts = pdf_loader.load_and_split(text_splitter) # type: ignore

            embeddings = OpenAIEmbeddings(model='text-embedding-3-small') # type: ignore
            vector = embeddings.embed_query('Test Documents')
            connection = psycopg2.connect(  # type: ignore
            dbname="testpgvector",
            user="postgres",
            password="Emids123",
            host="localhost",
            port="5432" )
            
            CONNECTION_STRING = 'postgresql+psycopg2://postgres:Emids123@localhost:5432/testpgvector' # type: ignore
            COLLECTION_NAME = os.path.splitext(uploaded_file.name)[0] # type: ignore
            #doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:1]])
          #  db = PGVector(connection_string=CONNECTION_STRING,embedding_function=embeddings,collection_name=COLLECTION_NAME)  # type: ignore
            db = PGVector.from_documents(embedding=embeddings,documents=texts,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING)

#print(len(texts))
#print(texts[0])      
# embedding=embeddings,documents=texts,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING)

#this is used to save documents
#db = PGVector.from_documents(embeddings=embeddings,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING,)
            
        
else:
    st.write("No file uploaded yet.")
