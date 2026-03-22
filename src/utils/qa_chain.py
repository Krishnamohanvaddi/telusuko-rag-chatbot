import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def build_qa_chain(vectorstore):
    """
    Builds and returns a LangChain RetrievalQA chain.

    Flow:
    user question
        → embed question
        → FAISS finds top 4 similar chunks
        → chunks + question sent to Groq LLM
        → answer returned
    """

    # Custom prompt: forces the LLM to answer only from document context
    prompt_template = """You are a helpful assistant that answers questions 
based ONLY on the provided context from uploaded PDF documents.

If the answer is not in the context, respond with:
"I don't have enough information in the uploaded documents to answer this."

Do not make up answers or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Groq LLM — free, cloud-hosted, no local GPU needed
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1   # low = factual, high = creative
    )

    # RetrievalQA chain: retrieves top 4 chunks then passes to LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # "stuff" = all chunks stuffed into one prompt
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}   # top 4 most relevant chunks
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain