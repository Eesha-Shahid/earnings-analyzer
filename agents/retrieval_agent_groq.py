"""
Retrieval Agent
---------------
Takes the QueryPlan from QueryUnderstandingAgent and:
1. Builds metadata filters from entities
2. Runs hybrid retrieval (semantic + BM25) against ChromaDB
3. Fetches structured data from yfinance when needed
4. Scores and re-ranks retrieved chunks
5. Self-corrects if retrieval quality is poor (re-queries with relaxed filters)
6. Returns ranked, deduplicated chunks with confidence scores
"""

import math
import time
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from db.chroma import chroma
from ingestion.processors.embedder import embedder
from ingestion.fetchers.yfinance_fetcher import YFinanceFetcher
from agents.query_agent_groq import QueryPlan
from config import Config

# ─── Output Schemas ────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    ticker: Optional[str] = None
    company: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None
    content_type: str = "text"         # "text" | "table" | "chart"
    source_url: Optional[str] = None
    filing_date: Optional[str] = None
    source: Optional[str] = None
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    time_decay_score: float = 0.0
    final_score: float = 0.0
    confidence: str = "medium"         # "high" | "medium" | "low"


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    yfinance_data: dict = Field(default_factory=dict)
    retrieval_quality: str = "good"    # "good" | "poor" | "empty"
    total_found: int = 0
    sources_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─── BM25 Helper ──────────────────────────────────────────────────────────────

class BM25Retriever:
    """Keyword-based retrieval to complement semantic search"""

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        tokenized_docs = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)

        # Clip negatives to 0 first, then normalize
        scores = [max(0.0, s) for s in scores]
        max_score = max(scores) if max(scores) > 0 else 1
        return [float(s / max_score) for s in scores]

# ─── Time Decay ───────────────────────────────────────────────────────────────

def compute_time_decay(filing_date: str, decay_rate: float = 0.002) -> float:
    """
    Newer documents score higher.
    score = e^(-decay * age_in_days)
    decay_rate=0.002 means a 1-year-old doc scores ~0.48
    """
    if not filing_date:
        return 0.5  # neutral if unknown

    try:
        filed = datetime.strptime(filing_date[:10], "%Y-%m-%d")
        age_days = (datetime.utcnow() - filed).days
        return math.exp(-decay_rate * age_days)
    except Exception:
        return 0.5


# ─── Metadata Filter Builder ──────────────────────────────────────────────────

def build_chroma_filter(plan: QueryPlan) -> Optional[dict]:
    """
    Build ChromaDB WHERE clause from QueryPlan entities.
    ChromaDB supports: $eq, $ne, $in, $and, $or
    """
    conditions = []

    # Ticker filter
    tickers = plan.retrieval_plan.tickers
    if tickers:
        if len(tickers) == 1:
            conditions.append({"ticker": {"$eq": tickers[0]}})
        else:
            conditions.append({"ticker": {"$in": tickers}})

    # Quarter filter — normalize "Q3 2024" → "Q3"
    quarters = []
    for q in plan.retrieval_plan.quarters:
        q_clean = q.strip().split()[0] if " " in q else q.strip()
        if q_clean.startswith("Q") and len(q_clean) == 2:
            quarters.append(q_clean)

    if quarters:
        if len(quarters) == 1:
            conditions.append({"quarter": {"$eq": quarters[0]}})
        else:
            conditions.append({"quarter": {"$in": quarters}})

    # Year filter
    years = plan.retrieval_plan.years
    if years:
        if len(years) == 1:
            conditions.append({"year": {"$eq": years[0]}})
        else:
            conditions.append({"year": {"$in": years}})

    # Section filter
    sections = [
        s for s in plan.retrieval_plan.sections
        if s in ["financial_results", "guidance", "risk_factors", "qa_session", "opening_remarks"]
    ]
    if sections:
        if len(sections) == 1:
            conditions.append({"section": {"$eq": sections[0]}})
        else:
            conditions.append({"section": {"$in": sections}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ─── Main Retrieval Agent ─────────────────────────────────────────────────────

class RetrievalAgent:

    # Weights for final score
    SEMANTIC_WEIGHT   = 0.50
    BM25_WEIGHT       = 0.30
    TIME_DECAY_WEIGHT = 0.20

    # Thresholds
    HIGH_CONFIDENCE   = 0.70
    LOW_CONFIDENCE    = 0.40
    MIN_CHUNKS        = 2       # if below this, trigger re-query

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.yfinance = YFinanceFetcher()

    # ── Core retrieval ────────────────────────────────────────────────────

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        where: Optional[dict],
    ) -> list[dict]:
        """Query ChromaDB with semantic embeddings + optional metadata filter"""
        try:
            results = chroma.query(
                query_text=query,
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            print(f"  ChromaDB query error: {e}")
            return []

        chunks = []
        ids       = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB cosine distance → similarity score
            semantic_score = max(0.0, 1.0 - dist)
            chunks.append({
                "chunk_id": doc_id,
                "text": doc,
                "metadata": meta,
                "semantic_score": semantic_score,
            })

        return chunks

    def _hybrid_score(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[RetrievedChunk]:
        """
        Combine semantic + BM25 + time decay into final score.
        """
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        bm25_scores = self.bm25.score(query, texts)

        scored = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            semantic = chunk["semantic_score"]
            bm25     = bm25_scores[i]
            decay    = compute_time_decay(meta.get("filing_date", ""))

            final = (
                self.SEMANTIC_WEIGHT   * semantic +
                self.BM25_WEIGHT       * bm25     +
                self.TIME_DECAY_WEIGHT * decay
            )

            if final >= self.HIGH_CONFIDENCE:
                confidence = "high"
            elif final >= self.LOW_CONFIDENCE:
                confidence = "medium"
            else:
                confidence = "low"

            scored.append(RetrievedChunk(
                chunk_id      = chunk["chunk_id"],
                text          = chunk["text"],
                ticker        = meta.get("ticker"),
                company       = meta.get("company"),
                quarter       = meta.get("quarter"),
                year          = meta.get("year"),
                section       = meta.get("section"),
                content_type  = meta.get("content_type", "text"),
                source_url    = meta.get("source_url"),
                filing_date   = meta.get("filing_date"),
                source        = meta.get("source"),
                semantic_score     = round(semantic, 4),
                bm25_score         = round(bm25, 4),
                time_decay_score   = round(decay, 4),
                final_score        = round(final, 4),
                confidence         = confidence,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored

    def _is_boilerplate(self, text: str) -> bool:
        text_lower = text.lower()
        boilerplate = [
            "safe harbor",
            "forward-looking statements",
            "actual results may differ",
            "risks and uncertainties",
            "cautionary note",
        ]
        return any(p in text_lower for p in boilerplate)

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks (same first 100 chars)"""
        seen = set()
        unique = []
        for chunk in chunks:
            key = chunk.text[:100].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique

    # ── Self-correction loop ──────────────────────────────────────────────

    def _retrieve_with_fallback(
        self,
        query: str,
        plan: QueryPlan,
        warnings: list[str],
    ) -> list[RetrievedChunk]:
        """
        Attempt 1: strict metadata filter
        Attempt 2: relax section filter
        Attempt 3: relax all filters (ticker only)
        Attempt 4: no filter at all
        """
        n_results = max(plan.retrieval_plan.n_results, 10)

        # Build filters for each attempt
        strict_filter  = build_chroma_filter(plan)

        # Relaxed — ticker only, no section/quarter constraints
        ticker_filter = None
        if plan.retrieval_plan.tickers:
            tickers = plan.retrieval_plan.tickers
            ticker_filter = (
                {"ticker": {"$eq": tickers[0]}}
                if len(tickers) == 1
                else {"ticker": {"$in": tickers}}
            )

        attempts = [
            ("strict",  strict_filter),
            ("relaxed", ticker_filter),
            ("no filter", None),
        ]

        for attempt_name, where_filter in attempts:
            print(f"  Retrieval attempt [{attempt_name}] filter={where_filter}")
            raw_chunks = self._semantic_search(query, n_results, where_filter)

            if not raw_chunks:
                warnings.append(f"No results with {attempt_name} filter, relaxing...")
                continue

            scored = self._hybrid_score(query, raw_chunks)
            unique = self._deduplicate(scored)

            # Check if we have enough good quality chunks
            good_chunks = [c for c in unique if c.confidence in ("high", "medium")]

            if len(good_chunks) >= self.MIN_CHUNKS:
                print(f"  ✅ Retrieved {len(unique)} chunks ({len(good_chunks)} good quality)")
                return unique

            warnings.append(
                f"Only {len(good_chunks)} good chunks with {attempt_name} filter, relaxing..."
            )

        # Return whatever we have even if low quality
        print(f"  ⚠️  Returning low-quality results after all attempts")
        return scored if scored else []

    # ── yfinance data fetch ────────────────────────────────────────────────

    def _fetch_yfinance(self, plan: QueryPlan) -> dict:
        """Fetch structured financial data when retrieval plan requires it"""
        if "yfinance_api" not in plan.retrieval_plan.sources_needed:
            return {}

        yf_data = {}
        for ticker in plan.retrieval_plan.tickers:
            try:
                print(f"  Fetching yfinance data for {ticker}...")
                yf_data[ticker] = {
                    "metrics": self.yfinance.get_financial_metrics(ticker),
                    "earnings": self.yfinance.get_earnings_history(ticker),
                    "info": self.yfinance.get_company_info(ticker),
                }
                time.sleep(0.5)  # avoid rate limiting
            except Exception as e:
                print(f"  yfinance error for {ticker}: {e}")
                yf_data[ticker] = {}

        return yf_data

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, plan: QueryPlan) -> RetrievalResult:
        print(f"\n--- Retrieval Agent ---")
        print(f"Query: {plan.original_query[:80]}")
        print(f"Tickers: {plan.retrieval_plan.tickers}")
        print(f"Sources: {plan.retrieval_plan.sources_needed}")

        warnings = []
        all_chunks = []
        sources_used = []

        # ── 1. Vector DB retrieval ─────────────────────────────────────────
        if "vector_db" in plan.retrieval_plan.sources_needed:
            # Run retrieval for each sub-question and merge
            queries_to_run = [plan.original_query]

            # Also run sub-questions for complex queries
            if plan.complexity == "complex" and len(plan.sub_questions) > 1:
                queries_to_run += [sq.question for sq in plan.sub_questions[:3]]

            seen_ids = set()
            for q in queries_to_run:
                chunks = self._retrieve_with_fallback(q, plan, warnings)
                for c in chunks:
                    if c.chunk_id not in seen_ids:
                        seen_ids.add(c.chunk_id)
                        all_chunks.append(c)

            # Re-sort merged chunks by final score
            all_chunks.sort(key=lambda x: x.final_score, reverse=True)

            # Cap at reasonable limit
            all_chunks = all_chunks[:20]

            if all_chunks:
                sources_used.append("vector_db")

        # ── 2. yfinance retrieval ─────────────────────────────────────────
        yf_data = {}
        if "yfinance_api" in plan.retrieval_plan.sources_needed:
            yf_data = self._fetch_yfinance(plan)
            if yf_data:
                sources_used.append("yfinance_api")

        # ── 3. Assess overall retrieval quality ────────────────────────────
        high_quality   = [c for c in all_chunks if c.confidence == "high"]
        medium_quality = [c for c in all_chunks if c.confidence in ("high", "medium")]

        if not all_chunks and not yf_data:
            quality = "empty"
            warnings.append("No data found from any source.")
        elif len(medium_quality) >= 3: 
            quality = "good"
        elif len(medium_quality) >= 1:
            quality = "poor"
            warnings.append("Low confidence retrieval — answers may be incomplete.")
        else:
            quality = "empty"
            warnings.append("No relevant data found.")

        return RetrievalResult(
            query=plan.original_query,
            chunks=all_chunks,
            yfinance_data=yf_data,
            retrieval_quality=quality,
            total_found=len(all_chunks),
            sources_used=sources_used,
            warnings=warnings,
        )


# Singleton
retrieval_agent = RetrievalAgent()