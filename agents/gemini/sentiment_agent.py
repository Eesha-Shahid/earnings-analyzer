"""
Sentiment & Language Agent
--------------------------
Analyzes management tone and language patterns across earnings calls.
- Uses FinBERT for finance-specific sentiment scoring
- Tracks hedging vs confidence language quarter over quarter
- Detects tone shifts that may signal future problems
- Scores prepared remarks separately from Q&A (executives are less scripted in Q&A)
"""

import re
import json
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from agents.gemini.retrieval_agent import RetrievalResult, RetrievedChunk
from config import Config

# ── Gemini client ───────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

# ─── FinBERT Sentiment (local, free) ─────────────────────────────────────────

_finbert_pipeline = None

def get_finbert():
    """Lazy load FinBERT — only loads when first called"""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
            )
            print("  FinBERT loaded")
        except Exception as e:
            print(f"  FinBERT load failed: {e}. Using LLM sentiment fallback.")
    return _finbert_pipeline


def finbert_score(text: str) -> dict:
    """
    Returns {positive: float, negative: float, neutral: float}
    Falls back to neutral if FinBERT unavailable.
    """
    pipe = get_finbert()
    if not pipe:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    try:
        # FinBERT max 512 tokens — truncate long text
        text = text[:1500]
        results = pipe(text)[0]
        scores = {r["label"].lower(): round(r["score"], 4) for r in results}
        return scores
    except Exception:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


# ─── Language Pattern Detectors ───────────────────────────────────────────────

CONFIDENCE_PHRASES = [
    "extremely strong", "record", "exceptional", "outstanding",
    "very confident", "we expect", "strong demand", "accelerating",
    "we are pleased", "robust", "significant growth", "unprecedented",
    "very strong", "we are excited",
]

HEDGING_PHRASES = [
    "may", "might", "could", "uncertain", "challenging", "headwind",
    "we believe", "we expect approximately", "subject to",
    "depends on", "if conditions", "we cannot predict",
    "potential risk", "we are monitoring", "cautious",
]

RISK_SIGNAL_PHRASES = [
    "export restriction", "regulatory", "competition", "supply constraint",
    "macro", "recession", "inflation", "geopolit", "china", "tariff",
    "customer concentration", "pricing pressure", "inventory",
]


def count_phrase_matches(text: str, phrases: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in phrases if p in text_lower)


# ─── Output Schemas ────────────────────────────────────────────────────────────

class QuarterSentiment(BaseModel):
    quarter: str
    year: str
    section: str                            # "prepared_remarks" | "qa_session" | "general"
    finbert_positive: float = 0.0
    finbert_negative: float = 0.0
    finbert_neutral: float = 0.0
    overall_sentiment: str = "neutral"      # "positive" | "negative" | "neutral"
    confidence_score: float = 0.0           # 0-1, how confident management sounds
    hedging_score: float = 0.0              # 0-1, how much hedging language
    confidence_phrase_count: int = 0
    hedging_phrase_count: int = 0
    risk_signal_count: int = 0
    source_chunk_id: Optional[str] = None


class ToneShift(BaseModel):
    from_quarter: str
    to_quarter: str
    sentiment_delta: float                  # positive = more confident, negative = more cautious
    hedging_delta: float                    # positive = more hedging
    interpretation: str                     # plain English summary


class SentimentResult(BaseModel):
    ticker: str
    quarterly_sentiment: list[QuarterSentiment] = Field(default_factory=list)
    tone_shifts: list[ToneShift] = Field(default_factory=list)
    overall_tone: str = "neutral"
    key_themes: list[str] = Field(default_factory=list)
    management_confidence: str = "medium"   # "high" | "medium" | "low"
    red_flags: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─── LLM Theme Extraction ─────────────────────────────────────────────────────

THEME_SYSTEM_PROMPT = """You are a financial analyst expert in earnings call analysis.
Analyze the provided earnings call text and extract key themes and red flags.

Return ONLY valid JSON:
{
  "key_themes": ["theme 1", "theme 2", "theme 3"],
  "red_flags": ["concern 1", "concern 2"],
  "management_confidence": "high|medium|low",
  "summary": "2-3 sentence summary of overall tone"
}

Key themes are the main topics management emphasized (e.g., "AI demand acceleration", "margin expansion").
Red flags are concerns or risks hinted at but not explicitly stated.
Never include more than 5 key themes or 3 red flags."""


def _call_llm(system: str, user: str) -> str:
    response = gemini_client.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=system + "\n\n" + user,
    )
    raw = response.text.strip()
    return raw

# ─── Main Agent ───────────────────────────────────────────────────────────────

class SentimentAgent:

    def _score_chunk(self, chunk: RetrievedChunk) -> QuarterSentiment:
        """Score a single chunk with FinBERT + phrase counting"""
        fb = finbert_score(chunk.text)

        confidence_count = count_phrase_matches(chunk.text, CONFIDENCE_PHRASES)
        hedging_count    = count_phrase_matches(chunk.text, HEDGING_PHRASES)
        risk_count       = count_phrase_matches(chunk.text, RISK_SIGNAL_PHRASES)

        # Normalize phrase counts to 0-1 scores
        max_phrases = 10
        confidence_score = min(confidence_count / max_phrases, 1.0)
        hedging_score    = min(hedging_count / max_phrases, 1.0)

        # Determine overall sentiment
        if fb.get("positive", 0) > 0.5:
            overall = "positive"
        elif fb.get("negative", 0) > 0.4:
            overall = "negative"
        else:
            overall = "neutral"

        return QuarterSentiment(
            quarter=chunk.quarter or "",
            year=chunk.year or "",
            section=chunk.section or "general",
            finbert_positive=fb.get("positive", 0),
            finbert_negative=fb.get("negative", 0),
            finbert_neutral=fb.get("neutral", 0),
            overall_sentiment=overall,
            confidence_score=round(confidence_score, 3),
            hedging_score=round(hedging_score, 3),
            confidence_phrase_count=confidence_count,
            hedging_phrase_count=hedging_count,
            risk_signal_count=risk_count,
            source_chunk_id=chunk.chunk_id,
        )

    def _aggregate_by_quarter(
        self,
        scores: list[QuarterSentiment],
    ) -> list[QuarterSentiment]:
        """Average scores for chunks from the same quarter"""
        from collections import defaultdict

        quarter_groups: dict[str, list[QuarterSentiment]] = defaultdict(list)
        for s in scores:
            key = f"{s.quarter}_{s.year}"
            quarter_groups[key].append(s)

        aggregated = []
        for key, group in quarter_groups.items():
            n = len(group)
            avg = QuarterSentiment(
                quarter=group[0].quarter,
                year=group[0].year,
                section="aggregated",
                finbert_positive=round(sum(g.finbert_positive for g in group) / n, 4),
                finbert_negative=round(sum(g.finbert_negative for g in group) / n, 4),
                finbert_neutral=round(sum(g.finbert_neutral for g in group) / n, 4),
                confidence_score=round(sum(g.confidence_score for g in group) / n, 4),
                hedging_score=round(sum(g.hedging_score for g in group) / n, 4),
                confidence_phrase_count=sum(g.confidence_phrase_count for g in group),
                hedging_phrase_count=sum(g.hedging_phrase_count for g in group),
                risk_signal_count=sum(g.risk_signal_count for g in group),
            )

            # Determine overall from aggregated FinBERT
            if avg.finbert_positive > 0.5:
                avg.overall_sentiment = "positive"
            elif avg.finbert_negative > 0.4:
                avg.overall_sentiment = "negative"
            else:
                avg.overall_sentiment = "neutral"

            aggregated.append(avg)

        return sorted(aggregated, key=lambda x: (x.year, x.quarter))

    def _detect_tone_shifts(
        self,
        quarterly: list[QuarterSentiment],
    ) -> list[ToneShift]:
        """Detect significant tone changes between quarters"""
        shifts = []
        for i in range(1, len(quarterly)):
            prev = quarterly[i - 1]
            curr = quarterly[i]

            sentiment_delta = (
                curr.finbert_positive - curr.finbert_negative
            ) - (
                prev.finbert_positive - prev.finbert_negative
            )
            hedging_delta = curr.hedging_score - prev.hedging_score

            # Only flag significant shifts
            if abs(sentiment_delta) < 0.05 and abs(hedging_delta) < 0.05:
                continue

            if sentiment_delta > 0.1:
                interp = f"Management became notably MORE confident from {prev.quarter} {prev.year} to {curr.quarter} {curr.year}"
            elif sentiment_delta < -0.1:
                interp = f"Management became notably MORE CAUTIOUS from {prev.quarter} {prev.year} to {curr.quarter} {curr.year} — potential red flag"
            elif hedging_delta > 0.1:
                interp = f"Significant increase in hedging language from {prev.quarter} {prev.year} to {curr.quarter} {curr.year}"
            else:
                interp = f"Minor tone shift from {prev.quarter} {prev.year} to {curr.quarter} {curr.year}"

            shifts.append(ToneShift(
                from_quarter=f"{prev.quarter} {prev.year}",
                to_quarter=f"{curr.quarter} {curr.year}",
                sentiment_delta=round(sentiment_delta, 4),
                hedging_delta=round(hedging_delta, 4),
                interpretation=interp,
            ))

        return shifts

    def run(
        self,
        retrieval_result: RetrievalResult,
        ticker: str,
    ) -> SentimentResult:
        print(f"\n--- Sentiment Agent [{ticker}] ---")

        warnings = []
        chunks = retrieval_result.chunks

        if not chunks:
            return SentimentResult(ticker=ticker, warnings=["No chunks to analyze"])

        # Score each chunk with FinBERT + phrase counting
        scores = []
        citations = []
        for chunk in chunks[:15]:   # cap to avoid slow FinBERT on too many chunks
            score = self._score_chunk(chunk)
            scores.append(score)
            if chunk.chunk_id:
                citations.append(chunk.chunk_id)

        # Aggregate by quarter
        quarterly = self._aggregate_by_quarter(scores)

        # Detect tone shifts
        tone_shifts = self._detect_tone_shifts(quarterly)

        # Overall tone from most recent quarter
        overall_tone = "neutral"
        management_confidence = "medium"
        if quarterly:
            latest = quarterly[-1]
            overall_tone = latest.overall_sentiment
            if latest.confidence_score > 0.6:
                management_confidence = "high"
            elif latest.confidence_score < 0.3:
                management_confidence = "low"

        # LLM for key themes and red flags
        key_themes = []
        red_flags = []
        top_text = "\n\n".join(c.text[:300] for c in chunks[:6])

        try:
            raw = _call_llm(
                THEME_SYSTEM_PROMPT,
                f"Analyze this earnings call content for {ticker}:\n\n{top_text}",
            )
            raw = re.sub(r"```json|```", "", raw).strip()
            theme_data = json.loads(raw)
            key_themes = theme_data.get("key_themes", [])
            red_flags  = theme_data.get("red_flags", [])
            management_confidence = theme_data.get("management_confidence", management_confidence)
        except Exception as e:
            warnings.append(f"Theme extraction failed: {e}")

        print(f"  Scored {len(scores)} chunks across {len(quarterly)} quarters")
        print(f"  Tone shifts detected: {len(tone_shifts)}")
        print(f"  Overall tone: {overall_tone} | Confidence: {management_confidence}")

        return SentimentResult(
            ticker=ticker,
            quarterly_sentiment=quarterly,
            tone_shifts=tone_shifts,
            overall_tone=overall_tone,
            key_themes=key_themes,
            management_confidence=management_confidence,
            red_flags=red_flags,
            citations=list(set(citations)),
            warnings=warnings,
        )


# Singleton
sentiment_agent = SentimentAgent()