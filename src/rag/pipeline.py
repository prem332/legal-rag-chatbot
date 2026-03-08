from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from src.rag.vectorstore import get_vectorstore
from src.rag.citations import extract_citations, format_citations
from src.rag.embeddings import get_embeddings
from src.memory.chat_history import get_or_create_session
from src.config.settings import settings
import mlflow
import tempfile
import os

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL,
    temperature=0
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

def ingest_documents(files: list) -> dict:
    all_docs = []

    for file_path, filename in files:
        loader = PyPDFLoader(file_path) if filename.endswith(".pdf") else TextLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = filename

        chunks = text_splitter.split_documents(docs)
        all_docs.extend(chunks)

    vectorstore, db_used = get_vectorstore()
    vectorstore.add_documents(all_docs)

    faiss_store = FAISS.from_documents(all_docs, get_embeddings())
    faiss_store.save_local(settings.FAISS_INDEX_PATH)

    return {
        "chunks_created": len(all_docs),
        "documents_processed": len(files),
        "vector_db": db_used
    }

def query_rag(question: str, session_id: str = None) -> dict:
    with mlflow.start_run(nested=True):
        mlflow.log_param("question", question[:100])
        mlflow.log_param("model", settings.GROQ_MODEL)

        session_id, memory = get_or_create_session(session_id)

        vectorstore, db_used = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.TOP_K,
                "score_threshold": 0.5
            }
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

        result = chain.invoke({"question": question})

        source_docs = result.get("source_documents", [])
        docs_with_scores = [(doc, 0.9) for doc in source_docs]
        citations = extract_citations(docs_with_scores)
        citation_text = format_citations(citations)

        answer = result["answer"] + citation_text

        mlflow.log_metric("citations_found", len(citations))
        mlflow.log_text(answer, "answer.txt")

        return {
            "answer": answer,
            "session_id": session_id,
            "citations": [c.__dict__ for c in citations],
            "vector_db_used": db_used
        }