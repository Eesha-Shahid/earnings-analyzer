import hashlib
from datetime import datetime
from ingestion.fetchers.edgar import EDGARFetcher
from ingestion.processors.chunker import chunk_document
from ingestion.processors.extractor import DocumentExtractor
from ingestion.processors.image_processor import ImageProcessor
from db.chroma import chroma
import yfinance as yf

class IngestionPipeline:
    def __init__(self):
        self.edgar = EDGARFetcher()
        self.extractor = DocumentExtractor()
        self.image_processor = ImageProcessor()

    def _make_doc_id(self, ticker: str, filing_date: str, form_type: str) -> str:
        raw = f"{ticker}_{filing_date}_{form_type}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _parse_quarter(self, filing_date: str) -> tuple[str, str]:
        """"8-K filings are filed after the quarter ends"""
        date = datetime.strptime(filing_date, "%Y-%m-%d")
        # Shift back 45 days to get the actual reported quarter
        from datetime import timedelta
        reported_date = date - timedelta(days=45)
        quarter = f"Q{(reported_date.month - 1) // 3 + 1}"
        return quarter, str(reported_date.year)
    
    def _already_ingested(self, doc_id: str) -> bool:
        try:
            result = chroma.collection.get(
                where={"doc_id": {"$eq": doc_id}},
                limit=1,
            )
            return len(result["ids"]) > 0
        except Exception:
            return False

    def ingest_ticker(self, ticker: str, limit: int = 8):
        print(f"\n{'='*50}")
        print(f"Ingesting {ticker}...")

        transcripts = self.edgar.fetch_transcript(ticker, limit=limit)
        stock = yf.Ticker(ticker)
        company_name = stock.info.get("longName", ticker)

        total_chunks = 0
        all_filing_chunks = []

        for filing in transcripts:
            doc_id = self._make_doc_id(
                ticker, filing["filing_date"], filing["form_type"]
            )

            if self._already_ingested(doc_id):
                print(f"  ⏭️  Skipping {ticker} {filing['filing_date']} — already ingested")
                continue

            quarter, year = self._parse_quarter(filing["filing_date"])
            base_metadata = {
                "ticker": ticker,
                "company": company_name,
                "form_type": filing["form_type"],
                "filing_date": filing["filing_date"],
                "quarter": quarter,
                "year": year,
                "source_url": filing["source_url"],
                "doc_id": doc_id,
            }

            # ── Extract structured content ──────────────────────────
            extracted = self.extractor.extract(
                filing["content"],
                filing["source_url"]
            )

            all_chunks = []

            # ── 1. Text chunks ──────────────────────────────────────
            full_text = "\n\n".join(extracted["text_blocks"])
            text_chunks = chunk_document(full_text, base_metadata, doc_id)
            all_chunks.extend(text_chunks)

            # ── 2. Table chunks ─────────────────────────────────────
            for t_idx, table in enumerate(extracted["tables"]):
                table_text = table.get("text") or table.get("data", "")
                if not table_text:
                    continue

                table_chunk = {
                    "id": f"{doc_id}_table_{t_idx}",
                    "text": table_text,
                    "metadata": {
                        **base_metadata,
                        "content_type": "table",
                        "section": "financial_data",
                        "page": table.get("page"),
                        "contains_numbers": True,
                        "chunk_index": t_idx,
                    }
                }
                all_chunks.append(table_chunk)

            # ── 3. Image/chart chunks ───────────────────────────────
            chart_descriptions = self.image_processor.process_images(
                extracted["images"],
                surrounding_text=full_text[:1000]
            )

            for c_idx, chart in enumerate(chart_descriptions):
                chart_chunk = {
                    "id": f"{doc_id}_chart_{c_idx}",
                    "text": chart["description"],
                    "metadata": {
                        **base_metadata,
                        "content_type": "chart",
                        "section": "financial_data",
                        "page": chart.get("page"),
                        "contains_numbers": True,
                        "chunk_index": c_idx,
                    }
                }
                all_chunks.append(chart_chunk)

            # ── Store everything ────────────────────────────────────
            if all_chunks:
            #     chroma.upsert(all_chunks)
                total_chunks += len(all_chunks)
                print(
                    f"  ✅ {ticker} {quarter} {year} — "
                    f"{len(text_chunks)} text, "
                    f"{len(extracted['tables'])} tables, "
                    f"{len(chart_descriptions)} charts"
                )
            all_filing_chunks.extend(all_chunks)

        if all_filing_chunks:
            chroma.upsert(all_filing_chunks)
            print(f"Done. Upserted {len(all_filing_chunks)} total chunks")
        return total_chunks