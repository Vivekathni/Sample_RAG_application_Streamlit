import streamlit as st

# Define page navigation
#page = st.sidebar.radio("Navigation", ["Home", "PDFUpload", "SearchInCollection"])

# Display content based on the selected page

st.title("Welcome to RAG Application")
st.write("Navigate to PDFUpload to upload a new PDF file")
st.write("Navigate to Search Collection to search from the uploaded file")