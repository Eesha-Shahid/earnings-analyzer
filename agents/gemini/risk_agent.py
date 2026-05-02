"""
Risk Extraction Agent
---------------------
Specifically hunts for risk language in earnings calls.
- Categorizes risks by type (regulatory, competitive, macro, operational)
- Tracks risk frequency and escalation across quarters
- Flags when a risk is mentioned more often than previous quarters
- Scores risk severity based on language intensity
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

# ─── Risk Categories ──────────────────────────────────────────────────────────

RISK_CATEGORIES = {
    "regulatory": [
        "export restriction", "regulation", "compliance", "government",
        "license", "ban", "sanction", "tariff", "trade restriction",
        "department of commerce", "entity list",
    ],
    "geopolitical": [
        "china", "taiwan", "geopolit", "russia", "middle east",
        "international tension", "trade war", "decoupling",
    ],
    "competitive": [
        "competition", "competitor", "market share", "pricing pressure",
        "alternative", "substitute", "amd", "intel", "google tpu",
        "in-house chip", "custom silicon",
    ],
    "supply_chain": [
        "supply", "manufacturing", "production", "shortage", "inventory",
        "lead time", "supplier", "tsmc", "foundry", "capacity constraint",
    ],
    "demand": [
        "demand", "customer", "order", "backlog", "cancellation",
        "push out", "slowdown", "softness", "weakness",
    ],
    "macro": [
        "macro", "recession", "inflation", "interest rate", "gdp",
        "economic slowdown", "uncertainty", "budget", "spending cuts",
    ],
    "operational": [
        "execution", "transition", "ramp", "yield", "quality",
        "delay", "timeline", "engineering", "headcount",
    ],
}

SEVERITY_KEYWORDS = {
    "high":   ["significant", "material", "major", "critical", "severe", "substantial"],
    "medium": ["moderate", "some", "certain", "potential", "may impact"],
    "low":    ["minor", "limited", "manageable", "small", "minimal"],
}


# ─── Output Schemas ────────────────────────────────────────────────────────────

class RiskItem(BaseModel):
    category: str
    description: str
    severity: str = "medium"           # "high" | "medium" | "low"
    quarter: Optional[str] = None
    year: Optional[str] = None
    mention_count: int = 1
    verbatim_excerpt: Optional[str] = None
    source_chunk_id: Optional[str] = None


class RiskTrend(BaseModel):
    category: str
    trend: str                          # "escalating" | "stable" | "improving"
    quarter_mentions: dict              # {"Q3 2024": 2, "Q4 2024": 5}
    interpretation: str


class RiskResult(BaseModel):
    ticker: str
    risks: list[RiskItem] = Field(default_factory=list)
    risk_trends: list[RiskTrend] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)
    escalating_risks: list[str] = Field(default_factory=list)
    overall_risk_level: str = "medium"  # "high" | "medium" | "low"
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─── LLM Risk Extraction ──────────────────────────────────────────────────────

RISK_SYSTEM_PROMPT = """You are a financial risk analyst specializing in earnings call analysis.
Extract all risks mentioned in the provided earnings call text.

Return ONLY valid JSON:
{
  "risks": [
    {
      "category": "regulatory|geopolitical|competitive|supply_chain|demand|macro|operational",
      "description": "clear description of the risk in 1-2 sentences",
      "severity": "high|medium|low",
      "quarter": "Q3",
      "year": "2024",
      "verbatim_excerpt": "exact quote from the text (max 100 words)",
      "chunk_id": "chunk id this came from"
    }
  ]
}

Rules:
- Only extract risks explicitly mentioned or clearly implied
- Do not fabricate risks not in the text
- Severity high = could materially impact financials
- Severity medium = worth monitoring
- Severity low = acknowledged but not concerning
- If no risks found return {"risks": []}"""


def _call_llm(system: str, user: str) -> str:
    response = gemini_client.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=system + "\n\n" + user,
    )
    raw = response.text.strip()
    return raw

# ─── Main Agent ───────────────────────────────────────────────────────────────

class RiskExtractionAgent:

    def _keyword_categorize(self, text: str) -> list[str]:
        """Fast keyword-based categorization before LLM"""
        text_lower = text.lower()
        matched = []
        for category, keywords in RISK_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                matched.append(category)
        return matched

    def _assess_severity(self, text: str) -> str:
        """Assess risk severity from language intensity"""
        text_lower = text.lower()
        for severity, keywords in SEVERITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return severity
        return "medium"

    def _extract_risks_from_chunks(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RiskItem]:
        """
        Two-pass extraction:
        Pass 1 — fast keyword scan on all chunks
        Pass 2 — LLM on risk-flagged chunks only
        """
        # Pass 1: keyword scan to find risk-relevant chunks
        risk_chunks = []
        for chunk in chunks:
            categories = self._keyword_categorize(chunk.text)
            if categories:
                risk_chunks.append((chunk, categories))

        if not risk_chunks:
            # Fall back to all chunks if no specific risk chunks found
            risk_chunks = [(c, ["general"]) for c in chunks[:6]]

        # Pass 2: LLM extraction on risk chunks only
        context_parts = []
        for chunk, categories in risk_chunks[:8]:
            context_parts.append(
                f"[CHUNK {chunk.chunk_id} | {chunk.ticker} {chunk.quarter} "
                f"{chunk.year} | categories={categories}]\n{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        try:
            raw = _call_llm(
                RISK_SYSTEM_PROMPT,
                f"Extract all risks from these earnings call sections:\n\n{context}",
            )
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
        except Exception as e:
            return []

        risks = []
        for r in data.get("risks", []):
            risks.append(RiskItem(
                category=r.get("category", "operational"),
                description=r.get("description", ""),
                severity=r.get("severity", "medium"),
                quarter=r.get("quarter"),
                year=r.get("year"),
                verbatim_excerpt=r.get("verbatim_excerpt"),
                source_chunk_id=r.get("chunk_id"),
            ))

        return risks

    def _compute_risk_trends(
        self,
        risks: list[RiskItem],
    ) -> list[RiskTrend]:
        """Track how often each risk category is mentioned across quarters"""
        from collections import defaultdict

        # Count mentions per category per quarter
        category_quarters: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for risk in risks:
            if risk.quarter and risk.year:
                q_key = f"{risk.quarter} {risk.year}"
                category_quarters[risk.category][q_key] += 1

        trends = []
        for category, quarter_counts in category_quarters.items():
            sorted_quarters = sorted(quarter_counts.keys())
            counts = [quarter_counts[q] for q in sorted_quarters]

            if len(counts) >= 2:
                if counts[-1] > counts[-2] * 1.5:
                    trend = "escalating"
                    interp = f"{category} risk is escalating — mentioned more frequently in recent quarters"
                elif counts[-1] < counts[-2] * 0.5:
                    trend = "improving"
                    interp = f"{category} risk appears to be improving — fewer mentions recently"
                else:
                    trend = "stable"
                    interp = f"{category} risk remains stable across quarters"
            else:
                trend = "stable"
                interp = f"{category} risk detected (insufficient quarters to determine trend)"

            trends.append(RiskTrend(
                category=category,
                trend=trend,
                quarter_mentions=dict(quarter_counts),
                interpretation=interp,
            ))

        return sorted(trends, key=lambda t: t.trend == "escalating", reverse=True)

    def _assess_overall_risk(self, risks: list[RiskItem]) -> str:
        if not risks:
            return "low"
        high_count = sum(1 for r in risks if r.severity == "high")
        if high_count >= 2:
            return "high"
        if high_count >= 1:
            return "medium"
        return "low"

    def run(
        self,
        retrieval_result: RetrievalResult,
        ticker: str,
    ) -> RiskResult:
        print(f"\n--- Risk Extraction Agent [{ticker}] ---")

        warnings = []
        chunks = retrieval_result.chunks

        if not chunks:
            return RiskResult(ticker=ticker, warnings=["No chunks to analyze"])

        risks = self._extract_risks_from_chunks(chunks)

        citations = list(set(
            r.source_chunk_id for r in risks if r.source_chunk_id
        ))

        risk_trends = self._compute_risk_trends(risks)

        escalating = [t.category for t in risk_trends if t.trend == "escalating"]

        top_risks = [
            r.description for r in
            sorted(risks, key=lambda x: (x.severity == "high", x.mention_count), reverse=True)
        ][:3]

        overall_risk = self._assess_overall_risk(risks)

        print(f"  Found {len(risks)} risks across {len(set(r.category for r in risks))} categories")
        print(f"  Escalating: {escalating}")
        print(f"  Overall risk level: {overall_risk}")

        return RiskResult(
            ticker=ticker,
            risks=risks,
            risk_trends=risk_trends,
            top_risks=top_risks,
            escalating_risks=escalating,
            overall_risk_level=overall_risk,
            citations=citations,
            warnings=warnings,
        )


# Singleton
risk_agent = RiskExtractionAgent()