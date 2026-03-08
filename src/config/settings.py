from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"

    # Embeddings
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector DB
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "legal-rag"
    PINECONE_ENV: str = "gcp-starter"
    USE_FAISS_FALLBACK: bool = True
    FAISS_INDEX_PATH: str = "faiss_index"

    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    # Memory
    MAX_HISTORY: int = 10

    class Config:
        env_file = ".env"

settings = Settings()