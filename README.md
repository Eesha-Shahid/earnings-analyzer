# Earnings Analyzer — Multi-Agent RAG Pipeline

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp .env.example .env
# Fill in your GEMINI_API_KEY in .env
```

## Run Ingestion

```bash
python run_ingestion.py
```

## Project Structure

```
earnings-analyzer/
├── config.py                          # All config and env vars
├── run_ingestion.py                   # Entry point for ingestion
├── requirements.txt
├── .env.example
├── ingestion/
│   ├── pipeline.py                    # Orchestrates full ingestion
│   ├── fetchers/
│   │   ├── edgar.py                   # SEC EDGAR fetcher
│   │   ├── yfinance_fetcher.py        # Structured financial data
│   │   └── motleyfool.py              # Backup transcript scraper
│   └── processors/
│       ├── chunker.py                 # Section-aware chunking
│       ├── embedder.py                # BAAI/bge-large-en embeddings
│       ├── extractor.py               # Text + table extraction
│       └── image_processor.py         # Chart → text via Gemini Vision
├── db/
│   └── chroma.py                      # ChromaDB client (singleton)
├── agents/                            # Coming next
├── api/                               # Coming next
└── evaluation/                        # Coming next
```

## Data Flow

```
SEC EDGAR / Motley Fool
        ↓
  DocumentExtractor
  (text, tables, images)
        ↓
  ImageProcessor
  (charts → text descriptions via Gemini Vision)
        ↓
  Chunker
  (section-aware splitting with metadata)
        ↓
  Embedder
  (BAAI/bge-large-en)
        ↓
  ChromaDB
  (persistent vector store)
```
