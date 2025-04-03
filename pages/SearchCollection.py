import streamlit as st
import psycopg2
import pandas as pd
from langchain_openai import OpenAIEmbeddings
openai = OpenAIEmbeddings(openai_api_key="sk-openaikey")
from dotenv import load_dotenv
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

st.write("Select the document and search ?")
# Define connection parameters
connection_params = {
    "host": "localhost",
    "database": "testpgvector",
    "user": "postgres",
    "password": "Emids123",
    "port": 5432  # default port
}

# Function to fetch data from PostgreSQL
def fetch_data_from_postgres():
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**connection_params)
    try:
        # Use a query to select the column you want to display in radio button options
        query = "SELECT name FROM public.langchain_pg_collection;"
        df = pd.read_sql_query(query, conn)
    finally:
        # Ensure the connection is closed
        conn.close()
    
    # Convert the column values to a list for Streamlit's radio button options
    return df['name'].tolist()

# Fetch data and bind to radio button
options = fetch_data_from_postgres()
selected_option = st.radio("Select an Option", options)

# Display selected option
st.write("You selected:", selected_option)
user_message = st.text_input("Your message:")

# Send button
if st.button("Send"):
      embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
#vector = embeddings.embed_query('Test Documents')
 #print(len(texts))
#print(texts[0])      
      CONNECTION_STRING = 'postgresql+psycopg2://postgres:Emids123@localhost:5432/testpgvector'
      COLLECTION_NAME = selected_option
#doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:1]])
      db = PGVector(connection_string=CONNECTION_STRING,embedding_function=embeddings,collection_name=COLLECTION_NAME) # embedding=embeddings,documents=texts,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING)

#this is used to save documents
#db = PGVector.from_documents(embeddings=embeddings,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING,)
#db = PGVector(embeddings=embeddings,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING,use_jsonb=True)

      results = db.similarity_search(query=user_message, k=5)

#print(results)

#print(results[1])

      chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# Create a template or prompt
      template = ChatPromptTemplate.from_messages(
                 [SystemMessage(content=results[0].page_content + results[1].page_content + results[2].page_content),
           HumanMessage(content=user_message)]
      )

# Send a request to OpenAI Chat Completion API
      response = chat(template.format_messages())

# Display the response
      st.write(response.content)
