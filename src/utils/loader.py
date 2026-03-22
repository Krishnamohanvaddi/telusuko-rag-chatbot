import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(uploaded_file) -> list:
    """
    Takes a Streamlit uploaded file object,
    saves it temporarily to disk, loads it with PyPDFLoader,
    and returns a list of LangChain Document objects.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)  # always delete temp file after loading

    return documents