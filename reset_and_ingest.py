"""
Clear ChromaDB and re-ingest fresh data.
Run: python reset_and_ingest.py
"""

import shutil
import os
from config import Config

# ── Step 1: Delete ChromaDB on disk ──────────────────────────────────────────
if os.path.exists(Config.CHROMA_PATH):
    shutil.rmtree(Config.CHROMA_PATH)
    print(f"✅ Deleted ChromaDB at {Config.CHROMA_PATH}")
else:
    print(f"No ChromaDB found at {Config.CHROMA_PATH}, starting fresh")

# ── Step 2: Re-ingest ─────────────────────────────────────────────────────────
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
pipeline.ingest_ticker("NVDA", limit=8)

# Add more tickers as needed:
# pipeline.ingest_all(["NVDA", "AAPL", "MSFT"])