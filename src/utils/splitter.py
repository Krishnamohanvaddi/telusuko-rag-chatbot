from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(documents: list) -> list:
    """
    Takes a list of LangChain Document objects,
    splits them into smaller chunks, and returns the chunks.

    chunk_size=1000   → each chunk is ~1000 characters
    chunk_overlap=200 → chunks overlap by 200 chars so
                        context is not lost at boundaries
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks