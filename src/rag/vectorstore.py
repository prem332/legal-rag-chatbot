from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from src.rag.embeddings import get_embeddings
from src.config.settings import settings
import os

embeddings = get_embeddings()

def get_pinecone_store():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    if settings.PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX,
        embedding=embeddings
    )

def get_faiss_store():
    if os.path.exists(settings.FAISS_INDEX_PATH):
        return FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def get_vectorstore():
    try:
        return get_pinecone_store(), "pinecone"
    except Exception as e:
        print(f"Pinecone failed: {e}. Using FAISS fallback.")
        faiss_store = get_faiss_store()
        if faiss_store:
            return faiss_store, "faiss"
        raise Exception("Both Pinecone and FAISS unavailable")