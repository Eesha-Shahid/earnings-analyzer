"""
Query Understanding Agent
-------------------------
First agent in the pipeline. Receives raw user query and outputs:
1. Extracted entities (ticker, company, dates, metrics)
2. Question type classification
3. Decomposed sub-questions
4. Retrieval plan (which sources, which filters)
"""

import json
import re
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq
from config import Config

client = Groq(api_key=Config.GROQ_API_KEY)

# ─── Output Schemas ────────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    tickers: list[str] = Field(default_factory=list)        # ["NVDA", "AMD"]
    companies: list[str] = Field(default_factory=list)      # ["Nvidia", "AMD"]
    quarters: list[str] = Field(default_factory=list)       # ["Q3", "Q4"]
    years: list[str] = Field(default_factory=list)          # ["2024", "2023"]
    metrics: list[str] = Field(default_factory=list)        # ["revenue", "gross margin"]
    time_range: Optional[str] = None                        # "last 4 quarters"


class SubQuestion(BaseModel):
    question: str
    ticker: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[str] = None
    metric: Optional[str] = None
    section: Optional[str] = None   # which doc section to look in


class RetrievalPlan(BaseModel):
    tickers: list[str]
    quarters: list[str]
    years: list[str]
    sections: list[str]             # ["financial_results", "guidance", "risk_factors"]
    sources_needed: list[str]       # ["vector_db", "yfinance_api"]
    n_results: int = 10             # how many chunks to retrieve
    requires_comparison: bool = False
    requires_calculation: bool = False


class QueryPlan(BaseModel):
    original_query: str
    question_type: str              # "educational" | "research" | "comparative" | "realtime"
    complexity: str                 # "simple" | "medium" | "complex"
    entities: ExtractedEntities
    sub_questions: list[SubQuestion]
    retrieval_plan: RetrievalPlan
    user_level: str                 # "beginner" | "intermediate" | "advanced"
    response_format: str            # "summary" | "detailed" | "comparison_table" | "bullet_points"


# ─── Known tickers map for fast lookup ─────────────────────────────────────────

KNOWN_TICKERS = {
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "tesla": "TSLA", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "amd": "AMD",
    "intel": "INTC", "qualcomm": "QCOM", "broadcom": "AVGO",
}

METRIC_KEYWORDS = [
    "revenue", "gross margin", "operating margin", "net income",
    "eps", "earnings per share", "ebitda", "free cash flow",
    "guidance", "outlook", "growth", "profit", "loss",
    "pricing", "average selling price", "asp", "cost",
    "headcount", "employees", "capex", "r&d", "debt", "cash",
]

SECTION_KEYWORDS = {
    "financial_results": ["revenue", "margin", "profit", "eps", "income", "growth"],
    "guidance": ["guidance", "outlook", "next quarter", "forecast", "expect"],
    "risk_factors": ["risk", "headwind", "challenge", "concern", "threat", "competition"],
    "qa_session": ["analyst", "question", "asked", "q&a"],
    "opening_remarks": ["prepared remarks", "overview", "highlights"],
}


# ─── Helper functions ──────────────────────────────────────────────────────────

def extract_tickers_fast(query: str) -> list[str]:
    """Fast regex-based ticker extraction before LLM call"""
    tickers = []

    # Match explicit uppercase tickers like NVDA, AAPL
    pattern = r'\b([A-Z]{1,5})\b'
    matches = re.findall(pattern, query)
    common_words = {
      "I", "A", "THE", "AND", "OR", "IN", "OF", "FOR", "IS", "IT", "TO",
      "PE", "EPS", "CEO", "CFO", "COO", "AI", "ML", "US", "EU", "UK",
      "Q1", "Q2", "Q3", "Q4", "YOY", "QOQ", "GDP", "IPO", "ETF", "API",
  }
    tickers.extend([m for m in matches if m not in common_words])

    # Match company names
    query_lower = query.lower()
    for name, ticker in KNOWN_TICKERS.items():
        if name in query_lower and ticker not in tickers:
            tickers.append(ticker)

    return list(set(tickers))


def detect_sections_needed(query: str) -> list[str]:
    """Determine which document sections are relevant to the query"""
    query_lower = query.lower()
    sections = []
    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            sections.append(section)
    return sections if sections else ["general"]


def detect_user_level(query: str) -> str:
    """Rough heuristic for user expertise level"""
    beginner_signals = ["what is", "what does", "explain", "how does", "what are", "define"]
    advanced_signals = ["basis points", "yoy", "qoq", "cagr", "ev/ebitda", "dcf", "var", "wacc"]

    query_lower = query.lower()
    if any(s in query_lower for s in beginner_signals):
        return "beginner"
    if any(s in query_lower for s in advanced_signals):
        return "advanced"
    return "intermediate"


# ─── Main Agent ───────────────────────────────────────────────────────────────

class QueryUnderstandingAgent:

    SYSTEM_PROMPT = """You are a financial query analysis expert.
      Your job is to analyze investor questions and return a structured JSON plan.

      You must return ONLY valid JSON matching this exact schema — no explanation, no markdown:
      {
        "question_type": "educational|research|comparative|realtime",
        "complexity": "simple|medium|complex",
        "entities": {
          "tickers": [],
          "companies": [],
          "quarters": [],
          "years": [],
          "metrics": [],
          "time_range": null
        },
        "sub_questions": [
          {
            "question": "string",
            "ticker": "string or null",
            "quarter": "string or null",
            "year": "string or null",
            "metric": "string or null",
            "section": "string or null"
          }
        ],
        "retrieval_plan": {
          "tickers": [],
          "quarters": [],
          "years": [],
          "sections": [],
          "sources_needed": [],
          "n_results": 10,
          "requires_comparison": false,
          "requires_calculation": false
        },
        "user_level": "beginner|intermediate|advanced",
        "response_format": "summary|detailed|comparison_table|bullet_points"
      }

      Question types:
      - educational: user wants to understand a concept ("what is gross margin")
      - research: user wants data on a specific company/period ("Nvidia Q3 2024 revenue")
      - comparative: user wants to compare ("Apple vs Microsoft cloud revenue")
      - realtime: user wants current market data ("what is Nvidia's stock price")

      Sources:
      - vector_db: for earnings transcripts, filings
      - yfinance_api: for real-time prices, ratios, current metrics
      - both: when question needs transcript context + actual numbers

      IMPORTANT: Generate a maximum of 5 sub-questions. Group related quarters together rather than creating one sub-question per quarter."""

    def run(self, query: str) -> QueryPlan:
      fast_tickers = extract_tickers_fast(query)
      sections_needed = detect_sections_needed(query)
      user_level = detect_user_level(query)

      prompt = f"""Analyze this investor query and return the JSON plan.

        Query: "{query}"

        Pre-extracted tickers (verify these): {fast_tickers}
        Detected sections: {sections_needed}
        Detected user level: {user_level}

        Return only JSON."""

      for attempt in range(3):
          try:
              response = client.chat.completions.create(
                  model=Config.GROQ_MODEL,
                  messages=[
                      {"role": "system", "content": self.SYSTEM_PROMPT},
                      {"role": "user", "content": prompt},
                  ],
                  temperature=0.1,       # low temp for consistent JSON
                  max_tokens=2048,
              )
              raw = response.choices[0].message.content.strip()
              raw = re.sub(r"```json|```", "", raw).strip()
              data = json.loads(raw)

              entities = ExtractedEntities(**data.get("entities", {}))
              for t in fast_tickers:
                  if t not in entities.tickers:
                      entities.tickers.append(t)

              sub_questions = [
                  SubQuestion(**sq) for sq in data.get("sub_questions", [])
              ]

              retrieval_plan = RetrievalPlan(**data.get("retrieval_plan", {
                  "tickers": entities.tickers,
                  "quarters": entities.quarters,
                  "years": entities.years,
                  "sections": sections_needed,
                  "sources_needed": ["vector_db"],
              }))

              return QueryPlan(
                  original_query=query,
                  question_type=data.get("question_type", "research"),
                  complexity=data.get("complexity", "medium"),
                  entities=entities,
                  sub_questions=sub_questions,
                  retrieval_plan=retrieval_plan,
                  user_level=data.get("user_level", user_level),
                  response_format=data.get("response_format", "detailed"),
              )

          except Exception as e:
              if "429" in str(e) or "rate_limit" in str(e).lower():
                  wait = 30 * (attempt + 1)
                  print(f"Rate limit hit. Waiting {wait}s... (attempt {attempt + 1}/3)")
                  import time
                  time.sleep(wait)
              else:
                  print(f"Query agent error: {e}. Using fallback plan.")
                  return self._fallback_plan(
                      query, fast_tickers, sections_needed, user_level
                  )

      return self._fallback_plan(query, fast_tickers, sections_needed, user_level)

    def _fallback_plan(
        self,
        query: str,
        tickers: list[str],
        sections: list[str],
        user_level: str,
    ) -> QueryPlan:
        """
        If LLM call fails or returns bad JSON,
        fall back to a basic plan using fast extraction.
        """
        entities = ExtractedEntities(tickers=tickers)
        return QueryPlan(
            original_query=query,
            question_type="research",
            complexity="medium",
            entities=entities,
            sub_questions=[SubQuestion(question=query, ticker=tickers[0] if tickers else None)],
            retrieval_plan=RetrievalPlan(
                tickers=tickers,
                quarters=[],
                years=[],
                sections=sections,
                sources_needed=["vector_db"],
            ),
            user_level=user_level,
            response_format="detailed",
        )


# Singleton
query_agent = QueryUnderstandingAgent()