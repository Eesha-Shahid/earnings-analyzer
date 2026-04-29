"""
Test the Retrieval Agent end-to-end.
Run: python test_retrieval_agent.py
"""

from agents.query_agent_groq import query_agent
from agents.retrieval_agent_groq import retrieval_agent

TEST_QUERIES = [
    "What was Nvidia's revenue in Q3 2024?",
    "What risks did Nvidia management mention about China exports?",
    "How has Nvidia's gross margin changed over the last 4 quarters?",
]

def print_result(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    # Step 1 — Query Understanding
    plan = query_agent.run(query)
    print(f"Plan: type={plan.question_type}, tickers={plan.entities.tickers}")

    # Step 2 — Retrieval
    result = retrieval_agent.run(plan)

    print(f"\nRetrieval Quality: {result.retrieval_quality}")
    print(f"Total Chunks:      {result.total_found}")
    print(f"Sources Used:      {result.sources_used}")

    if result.warnings:
        print(f"Warnings:")
        for w in result.warnings:
            print(f"  ⚠️  {w}")

    print(f"\nTop 5 Chunks:")
    for i, chunk in enumerate(result.chunks[:5], 1):
        print(f"\n  [{i}] Score: {chunk.final_score:.3f} ({chunk.confidence})")
        print(f"       Ticker: {chunk.ticker} | {chunk.quarter} {chunk.year}")
        print(f"       Section: {chunk.section} | Type: {chunk.content_type}")
        print(f"       Semantic: {chunk.semantic_score:.3f} | "
              f"BM25: {chunk.bm25_score:.3f} | "
              f"Decay: {chunk.time_decay_score:.3f}")
        print(f"       Text: {chunk.text[:150]}...")

    if result.yfinance_data:
        print(f"\nyFinance Data:")
        for ticker, data in result.yfinance_data.items():
            metrics = data.get("metrics", {})
            print(f"  {ticker}:")
            print(f"    Revenue TTM:    {metrics.get('revenue_ttm')}")
            print(f"    Gross Margin:   {metrics.get('gross_margin')}")
            print(f"    PE Ratio:       {metrics.get('pe_ratio')}")
            print(f"    Analyst Rec:    {metrics.get('analyst_recommendation')}")


if __name__ == "__main__":
    for query in TEST_QUERIES:
        print_result(query)