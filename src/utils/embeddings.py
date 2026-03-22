from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_embeddings():
    """
    Returns a HuggingFace embedding model.
    Model downloads automatically on first run (~90 MB, one-time only).
    Runs on CPU so no GPU is needed.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embeddings


def build_vectorstore(chunks: list):
    """
    Takes text chunks, embeds them, and stores them in a
    FAISS vector store. Returns the vectorstore object.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore