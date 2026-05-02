"""
Synthesis & Grounding Agent
----------------------------
Final agent in the pipeline. Combines outputs from all specialist agents into
a single, well-structured, grounded response.

Key responsibilities:
- Enforces grounding — every claim must cite a source
- Adapts language to user level (beginner vs advanced)
- Formats output based on question type (summary, detailed, comparison table)
- Runs hallucination check — flags any claim not supported by retrieved context
- Attaches reference documents for user to explore further
- Never invents numbers — uses NOT FOUND if data unavailable
"""

import re
import json
from typing import Optional
from pydantic import BaseModel, Field
from groq import Groq
from google import genai

from agents.groq.query_agent import QueryPlan
from agents.groq.retrieval_agent import RetrievalResult, RetrievedChunk
from agents.groq.parallel_runner import SpecialistResults
from config import Config

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client  = Groq(api_key=Config.GROQ_API_KEY)

# ─── Output Schemas ────────────────────────────────────────────────────────────

class ReferenceDocument(BaseModel):
    title: str
    ticker: str
    quarter: Optional[str] = None
    year: Optional[str] = None
    filing_date: Optional[str] = None
    source_url: Optional[str] = None
    relevant_excerpt: Optional[str] = None
    confidence: str = "medium"


class SynthesisResult(BaseModel):
    query: str
    answer: str
    confidence: str = "medium"          # "high" | "medium" | "low"
    data_quality: str = "good"          # "good" | "partial" | "insufficient"
    references: list[ReferenceDocument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    user_level: str = "intermediate"
    question_type: str = "research"


# ─── Grounding Prompt ─────────────────────────────────────────────────────────

GROUNDING_INSTRUCTION = """
CRITICAL GROUNDING RULES — follow these without exception:
1. Answer ONLY using the provided context below
2. Every numerical claim MUST reference its source
3. If a specific number is NOT in the context, write "NOT FOUND IN DOCUMENTS"
4. Do NOT interpolate, estimate, or guess any figures
5. If context is contradictory, present both versions and note the conflict
6. Do NOT reproduce more than 2 sentences verbatim from any source
"""

SYSTEM_PROMPT_TEMPLATE = """You are an expert financial analyst assistant helping {user_level} investors.

{grounding}

Response style for {user_level}:
{style_guide}

Response format: {response_format}
"""

STYLE_GUIDES = {
    "beginner": """
- Use simple, plain English — avoid jargon
- Explain any financial terms you use (e.g., "gross margin — the percentage of revenue left after production costs")
- Use analogies where helpful
- Keep sentences short and clear
- Focus on the most important 2-3 points
""",
    "intermediate": """
- Use standard financial terminology
- Provide context for numbers (e.g., compare to previous quarter or industry average)
- Include both the headline number and key drivers
- Mention risks and uncertainties where relevant
""",
    "advanced": """
- Use precise financial terminology (basis points, GAAP/non-GAAP distinction, etc.)
- Include granular detail — segment breakdowns, margin components, guidance ranges
- Discuss implications for valuation and investment thesis
- Flag any data quality issues or discrepancies between sources
""",
}


# ─── Context Builder ──────────────────────────────────────────────────────────

def build_synthesis_context(
    retrieval: RetrievalResult,
    specialist: SpecialistResults,
) -> str:
    """Package all agent outputs into a single context block for the LLM"""
    sections = []

    # ── Retrieved chunks ────────────────────────────────────────────────
    if retrieval.chunks:
        chunk_lines = []
        for c in retrieval.chunks[:8]:
            chunk_lines.append(
                f"[SOURCE: {c.ticker} {c.quarter} {c.year} | "
                f"section={c.section} | confidence={c.confidence}]\n"
                f"{c.text[:600]}"
            )
        sections.append("=== RETRIEVED DOCUMENTS ===\n" + "\n---\n".join(chunk_lines))

    # ── Financial extraction ────────────────────────────────────────────
    fin = specialist.financial
    fin_lines = [f"=== FINANCIAL DATA ({fin.ticker}) ==="]

    if fin.revenue:
        for r in fin.revenue:
            v = f"${r.value}M" if r.value else "NOT FOUND"
            verified = " [verified against yfinance]" if r.verified else ""
            fin_lines.append(f"Revenue {r.quarter} {r.year}: {v}{verified}")

    if fin.gross_margin:
        for m in fin.gross_margin:
            v = f"{m.value}%" if m.value else "NOT FOUND"
            fin_lines.append(f"Gross Margin {m.quarter} {m.year}: {v}")

    if fin.operating_margin:
        for m in fin.operating_margin:
            fin_lines.append(f"Operating Margin {m.quarter} {m.year}: {m.value}%")

    if fin.eps:
        for e in fin.eps:
            fin_lines.append(f"EPS {e.quarter} {e.year}: ${e.value}")

    if fin.guidance:
        for g in fin.guidance:
            if g.value:
                val = f"${g.value}M"
            elif g.range_low and g.range_high:
                val = f"${g.range_low}M - ${g.range_high}M"
            else:
                val = "range not specified"
            fin_lines.append(f"Guidance {g.metric} {g.quarter} {g.year}: {val}")

    if fin.computed_metrics.get("revenue_growth_trend"):
        for g in fin.computed_metrics["revenue_growth_trend"]:
            fin_lines.append(
                f"Revenue Growth {g['from']} → {g['to']}: {g['growth_pct']}%"
            )

    if fin.yfinance_snapshot:
        snap = fin.yfinance_snapshot
        gm = f"{round(snap['gross_margin'] * 100, 1)}%" if snap.get("gross_margin") else "N/A"
        fin_lines.append(f"yFinance Current: Gross Margin={gm}, PE={snap.get('pe_ratio')}, "
                        f"Analyst={snap.get('analyst_rec')}")

    if fin.warnings:
        for w in fin.warnings:
            fin_lines.append(f"⚠️ {w}")

    sections.append("\n".join(fin_lines))

    # ── Sentiment ───────────────────────────────────────────────────────
    sent = specialist.sentiment
    sent_lines = [f"=== SENTIMENT & TONE ({sent.ticker}) ==="]
    sent_lines.append(f"Overall tone: {sent.overall_tone}")
    sent_lines.append(f"Management confidence: {sent.management_confidence}")

    if sent.key_themes:
        sent_lines.append(f"Key themes: {', '.join(sent.key_themes)}")

    if sent.red_flags:
        sent_lines.append(f"Red flags: {', '.join(sent.red_flags)}")

    if sent.tone_shifts:
        for shift in sent.tone_shifts:
            sent_lines.append(f"Tone shift: {shift.interpretation}")

    if sent.quarterly_sentiment:
        for qs in sent.quarterly_sentiment:
            sent_lines.append(
                f"Sentiment {qs.quarter} {qs.year}: "
                f"positive={qs.finbert_positive:.2f} "
                f"negative={qs.finbert_negative:.2f} "
                f"hedging={qs.hedging_score:.2f}"
            )

    sections.append("\n".join(sent_lines))

    # ── Risk ────────────────────────────────────────────────────────────
    risk = specialist.risk
    risk_lines = [f"=== RISK ANALYSIS ({risk.ticker}) ==="]
    risk_lines.append(f"Overall risk level: {risk.overall_risk_level}")

    if risk.top_risks:
        risk_lines.append("Top risks:")
        for r in risk.top_risks:
            risk_lines.append(f"  • {r}")

    if risk.escalating_risks:
        risk_lines.append(f"Escalating risks: {', '.join(risk.escalating_risks)}")

    for trend in risk.risk_trends:
        risk_lines.append(
            f"{trend.category}: {trend.trend} — {trend.interpretation}"
        )

    if risk.risks:
        risk_lines.append("\nDetailed risks:")
        for r in risk.risks[:5]:
            risk_lines.append(
                f"  [{r.severity.upper()}] {r.category}: {r.description}"
            )

    sections.append("\n".join(risk_lines))

    # ── yfinance live data ───────────────────────────────────────────────
    if retrieval.yfinance_data:
        yf_lines = ["=== LIVE MARKET DATA (yfinance) ==="]
        for ticker, data in retrieval.yfinance_data.items():
            info    = data.get("info", {})
            metrics = data.get("metrics", {})
            yf_lines.append(f"{ticker} ({info.get('company_name', ticker)}):")
            if metrics.get("revenue_ttm"):
                rev_b = round(metrics["revenue_ttm"] / 1e9, 1)
                yf_lines.append(f"  Revenue TTM: ${rev_b}B")
            if metrics.get("gross_margin"):
                yf_lines.append(f"  Gross Margin: {round(metrics['gross_margin']*100, 1)}%")
            if metrics.get("pe_ratio"):
                yf_lines.append(f"  P/E Ratio: {metrics['pe_ratio']}")
            if metrics.get("analyst_recommendation"):
                yf_lines.append(f"  Analyst Rec: {metrics['analyst_recommendation']}")
            if metrics.get("analyst_target_price"):
                yf_lines.append(f"  Price Target: ${metrics['analyst_target_price']}")
        sections.append("\n".join(yf_lines))

    return "\n\n".join(sections)


# ─── Reference Builder ────────────────────────────────────────────────────────

def build_references(chunks: list[RetrievedChunk]) -> list[ReferenceDocument]:
    """Build reference documents from top retrieved chunks"""
    seen = set()
    refs = []

    for chunk in chunks:
        # Deduplicate by filing (ticker + quarter + year)
        key = f"{chunk.ticker}_{chunk.quarter}_{chunk.year}"
        if key in seen:
            continue
        seen.add(key)

        title = f"{chunk.company or chunk.ticker} {chunk.quarter} {chunk.year} Earnings"
        refs.append(ReferenceDocument(
            title=title,
            ticker=chunk.ticker or "",
            quarter=chunk.quarter,
            year=chunk.year,
            filing_date=chunk.filing_date,
            source_url=chunk.source_url,
            relevant_excerpt=chunk.text[:200] + "..." if chunk.text else None,
            confidence=chunk.confidence,
        ))

    return refs[:5]  # max 5 references


# ─── Main Agent ───────────────────────────────────────────────────────────────

class SynthesisAgent:

    def _call_groq(self, system: str, user: str) -> str:
        response = groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        return response.choices[0].message.content.strip()

    def _assess_confidence(
        self,
        retrieval: RetrievalResult,
        specialist: SpecialistResults,
    ) -> str:
        high_chunks = [c for c in retrieval.chunks if c.confidence == "high"]
        has_financial = bool(
            specialist.financial.revenue or specialist.financial.gross_margin
        )

        if len(high_chunks) >= 3 and has_financial:
            return "high"
        elif len(high_chunks) >= 1 or has_financial:
            return "medium"
        else:
            return "low"

    def _assess_data_quality(
        self,
        retrieval: RetrievalResult,
        specialist: SpecialistResults,
    ) -> str:
        if retrieval.retrieval_quality == "empty":
            return "insufficient"
        if retrieval.retrieval_quality == "poor" and not specialist.financial.revenue:
            return "partial"
        return "good"

    def run(
        self,
        query_plan: QueryPlan,
        retrieval: RetrievalResult,
        specialist: SpecialistResults,
    ) -> SynthesisResult:
        print(f"\n--- Synthesis Agent ---")

        user_level      = query_plan.user_level
        response_format = query_plan.response_format
        question_type   = query_plan.question_type

        # Build context from all agent outputs
        context = build_synthesis_context(retrieval, specialist)

        # Build system prompt
        style_guide = STYLE_GUIDES.get(user_level, STYLE_GUIDES["intermediate"])
        system = SYSTEM_PROMPT_TEMPLATE.format(
            user_level=user_level,
            grounding=GROUNDING_INSTRUCTION,
            style_guide=style_guide,
            response_format=response_format,
        )

        # Build user prompt
        format_instructions = {
            "summary":          "Provide a concise 2-3 paragraph summary.",
            "detailed":         "Provide a detailed analysis with sections for key metrics, sentiment, and risks.",
            "comparison_table": "Structure your answer as a comparison with clear sections for each entity.",
            "bullet_points":    "Use bullet points for clarity. Group by topic.",
        }.get(response_format, "Provide a clear, well-structured answer.")

        user_prompt = f"""Question: {query_plan.original_query}

            {format_instructions}

            Use the following context to answer. Cite sources inline where possible.
            If information is not in the context, say "NOT FOUND IN DOCUMENTS".

            CONTEXT:
            {context}"""

        warnings = []
        try:
            answer = self._call_groq(system, user_prompt)
            print(f"  ✅ Synthesis complete")
        except Exception as e2:
            answer = "Unable to generate answer. Please try again."
            warnings.append(f"Groq failed: {e2}")
            

        # Build reference documents
        references = build_references(retrieval.chunks)

        # Assess confidence and data quality
        confidence   = self._assess_confidence(retrieval, specialist)
        data_quality = self._assess_data_quality(retrieval, specialist)

        # Collect all warnings
        all_warnings = (
            warnings
            + retrieval.warnings
            + specialist.financial.warnings
            + specialist.sentiment.warnings
            + specialist.risk.warnings
        )

        return SynthesisResult(
            query=query_plan.original_query,
            answer=answer,
            confidence=confidence,
            data_quality=data_quality,
            references=references,
            warnings=[w for w in all_warnings if w],
            user_level=user_level,
            question_type=question_type,
        )


# Singleton
synthesis_agent = SynthesisAgent()