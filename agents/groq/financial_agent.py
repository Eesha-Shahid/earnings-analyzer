"""
Financial Extraction Agent
--------------------------
Extracts and cross-verifies numerical financial data from retrieved chunks.
- Pulls specific metrics (revenue, margins, EPS, guidance) from text/tables
- Cross-checks against yfinance structured data
- Computes derived metrics (growth rates, surprises) mathematically
- Never asks LLM to do arithmetic — always computes programmatically
- Returns structured Pydantic output with source citations
"""

import re
import json
from typing import Optional
from pydantic import BaseModel, Field
from groq import Groq
from agents.groq.retrieval_agent import RetrievalResult, RetrievedChunk
from config import Config

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=Config.GROQ_API_KEY)

# ─── Output Schemas ────────────────────────────────────────────────────────────

class QuarterlyMetric(BaseModel):
    quarter: str
    year: str
    value: Optional[float] = None
    unit: str = "millions_usd"
    source_chunk_id: Optional[str] = None
    verified: bool = False          # True if cross-checked against yfinance
    verification_delta_pct: Optional[float] = None


class GuidanceData(BaseModel):
    quarter: Optional[str] = None
    year: Optional[str] = None
    metric: str
    value: Optional[float] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    unit: str = "millions_usd"
    source_chunk_id: Optional[str] = None


class FinancialExtractionResult(BaseModel):
    ticker: str
    revenue: list[QuarterlyMetric] = Field(default_factory=list)
    gross_margin: list[QuarterlyMetric] = Field(default_factory=list)
    operating_margin: list[QuarterlyMetric] = Field(default_factory=list)
    net_income: list[QuarterlyMetric] = Field(default_factory=list)
    eps: list[QuarterlyMetric] = Field(default_factory=list)
    guidance: list[GuidanceData] = Field(default_factory=list)
    computed_metrics: dict = Field(default_factory=dict)
    yfinance_snapshot: dict = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─── Financial Calculator — never use LLM for math ───────────────────────────

class FinancialCalculator:

    @staticmethod
    def yoy_growth(current: float, previous: float) -> Optional[float]:
        if previous and previous != 0:
            return round((current - previous) / abs(previous) * 100, 2)
        return None

    @staticmethod
    def gross_margin(revenue: float, cogs: float) -> Optional[float]:
        if revenue and revenue != 0:
            return round((revenue - cogs) / revenue * 100, 2)
        return None

    @staticmethod
    def eps_surprise(actual: float, estimate: float) -> Optional[float]:
        if estimate and estimate != 0:
            return round((actual - estimate) / abs(estimate) * 100, 2)
        return None

    @staticmethod
    def normalize_value(value_str: str) -> Optional[float]:
        """Convert '26.3 billion' or '$26.3B' to float in millions"""
        if not value_str:
            return None

        value_str = value_str.strip().replace(",", "")
        multipliers = {"B": 1000, "M": 1, "K": 0.001,
                       "billion": 1000, "million": 1, "thousand": 0.001}

        # Match patterns like $26.3B, 26.3 billion, 26,300M
        pattern = r"[\$]?([\d.]+)\s*(B|M|K|billion|million|thousand)?"
        match = re.search(pattern, value_str, re.I)
        if not match:
            return None

        number = float(match.group(1))
        unit = match.group(2) or "M"

        for key, multiplier in multipliers.items():
            if unit.lower().startswith(key.lower()):
                return round(number * multiplier, 2)

        return round(number, 2)

    @staticmethod
    def verify_against_yfinance(
        extracted: float,
        yf_value: float,
        tolerance_pct: float = 5.0,
    ) -> tuple[bool, Optional[float]]:
        """Returns (is_match, delta_pct)"""
        if not extracted or not yf_value:
            return False, None

        delta = abs(extracted - yf_value) / abs(yf_value) * 100
        return delta <= tolerance_pct, round(delta, 2)


calc = FinancialCalculator()


# ─── LLM Extraction ───────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a financial data extraction expert.
Extract specific numerical financial metrics from the provided text chunks.

Return ONLY valid JSON in this exact format:
{
  "extractions": [
    {
      "metric": "revenue|gross_margin|operating_margin|net_income|eps|guidance",
      "quarter": "Q1|Q2|Q3|Q4",
      "year": "2024",
      "value": 26300.0,
      "unit": "millions_usd|percentage|per_share",
      "raw_text": "exact text the number came from",
      "chunk_id": "chunk id this came from",
      "confidence": "high|medium|low"
    }
  ],
  "guidance": [
    {
      "metric": "revenue|gross_margin",
      "quarter": "Q1",
      "year": "2025",
      "value": null,
      "range_low": 37500.0,
      "range_high": 38500.0,
      "unit": "millions_usd",
      "raw_text": "exact text"
    }
  ]
}

Rules:
- Convert all values to millions USD (e.g. $26.3B = 26300.0)
- For percentages keep as percentage (e.g. 74.6% = 74.6)
- If a number is not clearly stated, do not guess — omit it
- Extract guidance separately even if mentioned in the same chunk
- Return empty lists if nothing found, never null

Additional rules:
- "Revenue" means TOTAL revenue unless explicitly labeled as segment revenue
- Do NOT extract COGS, operating expenses, or cost items as revenue
- If a table shows multiple revenue lines, extract the TOTAL row only
- Guidance values must have explicit forward-looking language ("we expect", "outlook", "guidance")"""


def _call_llm(system: str, user: str) -> str:
    response = groq_client.chat.completions.create(
        model=Config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ─── Main Agent ───────────────────────────────────────────────────────────────

class FinancialExtractionAgent:

    def _prepare_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks for LLM context — only financial chunks"""
        financial_chunks = [
            c for c in chunks
            if c.section in ("financial_results", "guidance")
            or c.content_type in ("table", "chart")
        ]

        if not financial_chunks:
            financial_chunks = chunks[:8]  # fallback to top chunks

        lines = []
        for c in financial_chunks[:10]:  # cap at 10 to avoid token limits
            lines.append(
                f"[CHUNK {c.chunk_id} | {c.ticker} {c.quarter} {c.year} "
                f"| section={c.section}]\n{c.text}\n"
            )

        return "\n---\n".join(lines)

    def _parse_llm_output(
        self,
        raw: str,
        ticker: str,
    ) -> tuple[list[QuarterlyMetric], list[QuarterlyMetric],
               list[QuarterlyMetric], list[QuarterlyMetric],
               list[QuarterlyMetric], list[GuidanceData], list[str]]:

        raw = re.sub(r"```json|```", "", raw).strip()
        citations = []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return [], [], [], [], [], [], [f"JSON parse error: {e}"]

        revenue, gross_margin, op_margin, net_income, eps = [], [], [], [], []
        guidance_list = []

        for ex in data.get("extractions", []):
            metric = ex.get("metric", "").lower()
            chunk_id = ex.get("chunk_id", "")
            if chunk_id:
                citations.append(chunk_id)

            qm = QuarterlyMetric(
                quarter=ex.get("quarter", ""),
                year=ex.get("year", ""),
                value=ex.get("value"),
                unit=ex.get("unit") or "millions_usd",
                source_chunk_id=chunk_id,
            )

            if "revenue" in metric:
                revenue.append(qm)
            elif "gross_margin" in metric or "gross margin" in metric:
                gross_margin.append(qm)
            elif "operating_margin" in metric or "operating margin" in metric:
                op_margin.append(qm)
            elif "net_income" in metric or "net income" in metric:
                net_income.append(qm)
            elif "eps" in metric:
                eps.append(qm)

        for g in data.get("guidance", []):
            guidance_list.append(GuidanceData(
                metric=g.get("metric", ""),
                quarter=g.get("quarter"),
                year=g.get("year"),
                value=g.get("value"),
                range_low=g.get("range_low"),
                range_high=g.get("range_high"),
                unit=g.get("unit", "millions_usd"),
                source_chunk_id=g.get("chunk_id"),
            ))

        return revenue, gross_margin, op_margin, net_income, eps, guidance_list, citations

    def _cross_verify(
        self,
        metrics: list[QuarterlyMetric],
        metric_name: str,
        yf_data: dict,
        ticker: str,
    ) -> list[QuarterlyMetric]:
        """Cross-check extracted metrics against yfinance where possible"""
        yf_ticker = yf_data.get(ticker, {})
        yf_metrics = yf_ticker.get("metrics", {})

        yf_map = {
            "revenue":          yf_metrics.get("revenue_ttm"),
            "gross_margin":     yf_metrics.get("gross_margin"),
            "operating_margin": yf_metrics.get("operating_margin"),
        }

        yf_value = yf_map.get(metric_name)
        if not yf_value:
            return metrics

        # Only verify the most recent quarter
        if metrics:
            most_recent = sorted(
                metrics,
                key=lambda m: (m.year or "", m.quarter or ""),
                reverse=True,
            )[0]

            # Convert yfinance margin from decimal to percentage
            if metric_name in ("gross_margin", "operating_margin"):
                yf_value = yf_value * 100

            match, delta = calc.verify_against_yfinance(
                most_recent.value, yf_value
            )
            most_recent.verified = match
            most_recent.verification_delta_pct = delta

        return metrics

    def _compute_growth_rates(
        self,
        revenue: list[QuarterlyMetric],
    ) -> dict:
        """Compute QoQ and YoY growth from extracted revenue"""
        computed = {}
        if len(revenue) < 2:
            return computed
        
        # Deduplicate — keep highest value per quarter (most likely total revenue)
        quarter_map = {}
        for r in revenue:
            if r.value:
                key = f"{r.quarter}_{r.year}"
                if key not in quarter_map or r.value > quarter_map[key].value:
                    quarter_map[key] = r

        sorted_rev = sorted(
            [r for r in revenue if r.value],
            key=lambda m: (m.year or "", m.quarter or ""),
        )

        growth_rates = []
        for i in range(1, len(sorted_rev)):
            prev = sorted_rev[i - 1]
            curr = sorted_rev[i]
            if prev.value and curr.value:
                growth = calc.yoy_growth(curr.value, prev.value)
                growth_rates.append({
                    "from": f"{prev.quarter} {prev.year}",
                    "to":   f"{curr.quarter} {curr.year}",
                    "growth_pct": growth,
                })

        computed["revenue_growth_trend"] = growth_rates
        return computed

    def run(
        self,
        retrieval_result: RetrievalResult,
        ticker: str,
    ) -> FinancialExtractionResult:
        print(f"\n--- Financial Extraction Agent [{ticker}] ---")

        warnings = []
        context = self._prepare_context(retrieval_result.chunks)

        if not context.strip():
            warnings.append("No financial chunks available for extraction")
            return FinancialExtractionResult(
                ticker=ticker,
                warnings=warnings,
            )

        user_prompt = f"""Extract all financial metrics from these chunks for {ticker}:

            IMPORTANT: The query is about {ticker}. Focus on extracting the CURRENT QUARTER's data 
            as the PRIMARY result, not the comparison periods shown in tables.
            Tables often show multiple periods — extract ALL of them but clearly label each quarter/year.

            {context}

            For each number extracted, specify EXACTLY which quarter and year it belongs to based 
            on the column headers or surrounding text. Do not assume — read the headers carefully."""

        try:
            raw = _call_llm(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            raw = re.sub(r"```json|```", "", raw).strip()
        except Exception as e:
            warnings.append(f"LLM call failed: {e}")
            return FinancialExtractionResult(ticker=ticker, warnings=warnings)

        revenue, gross_margin, op_margin, net_income, eps, guidance, citations = \
            self._parse_llm_output(raw, ticker)

        # Cross-verify against yfinance
        yf_data = retrieval_result.yfinance_data
        revenue      = self._cross_verify(revenue, "revenue", yf_data, ticker)
        gross_margin = self._cross_verify(gross_margin, "gross_margin", yf_data, ticker)
        op_margin    = self._cross_verify(op_margin, "operating_margin", yf_data, ticker)

        # Compute derived metrics mathematically
        computed = self._compute_growth_rates(revenue)

        # yfinance snapshot for reference
        yf_snapshot = {}
        if ticker in yf_data:
            m = yf_data[ticker].get("metrics", {})
            yf_snapshot = {
                "revenue_ttm":          m.get("revenue_ttm"),
                "gross_margin":         m.get("gross_margin"),
                "operating_margin":     m.get("operating_margin"),
                "pe_ratio":             m.get("pe_ratio"),
                "analyst_target":       m.get("analyst_target_price"),
                "analyst_rec":          m.get("analyst_recommendation"),
            }

        print(f"  Extracted: {len(revenue)} revenue, {len(gross_margin)} margins, "
              f"{len(guidance)} guidance items")

        return FinancialExtractionResult(
            ticker=ticker,
            revenue=revenue,
            gross_margin=gross_margin,
            operating_margin=op_margin,
            net_income=net_income,
            eps=eps,
            guidance=guidance,
            computed_metrics=computed,
            yfinance_snapshot=yf_snapshot,
            citations=list(set(citations)),
            warnings=warnings,
        )


# Singleton
financial_agent = FinancialExtractionAgent()