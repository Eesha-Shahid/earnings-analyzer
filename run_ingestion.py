from ingestion.pipeline import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline()

    # Ingest a single ticker
    pipeline.ingest_ticker("NVDA", limit=8)

    # Or ingest multiple tickers
    # pipeline.ingest_all(["NVDA", "AAPL", "MSFT"])
