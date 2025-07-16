import os
import tempfile

from langchain.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document

# Accepts a Streamlit uploaded file and returns a LangChain Document object
def load_document(uploaded_file) -> list[Document]:
    # Create a temporary file to save the uploaded file
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Detect file type and use appropriate loader
    if suffix == ".pdf":
        loader = PyMuPDFLoader(file_path=tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path=tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path=tmp_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")

    documents = loader.load()

    return documents
