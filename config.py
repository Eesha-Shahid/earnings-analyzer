from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # LLM
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-1.5-flash"

    # ChromaDB
    CHROMA_PATH = "./chroma_db"
    CHROMA_COLLECTION = "earnings_calls"

    # PostgreSQL
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Embeddings
    EMBEDDING_MODEL = "BAAI/bge-large-en"

    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Data
    DATA_DIR = "./data/raw"
    SUPPORTED_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL"]
