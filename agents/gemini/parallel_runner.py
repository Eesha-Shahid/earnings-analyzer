"""
Parallel Agent Runner
---------------------
Runs Financial, Sentiment, and Risk agents simultaneously using threads.
All three agents are independent — no shared state — so parallelism is safe.
Results are collected and passed to the Synthesis Agent.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from agents.gemini.retrieval_agent import RetrievalResult
from agents.gemini.financial_agent import FinancialExtractionAgent, FinancialExtractionResult
from agents.gemini.sentiment_agent import SentimentAgent, SentimentResult
from agents.gemini.risk_agent import RiskExtractionAgent, RiskResult


class SpecialistResults(BaseModel):
    financial: FinancialExtractionResult
    sentiment: SentimentResult
    risk: RiskResult
    elapsed_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class ParallelAgentRunner:

    def __init__(self):
        self.financial_agent = FinancialExtractionAgent()
        self.sentiment_agent = SentimentAgent()
        self.risk_agent      = RiskExtractionAgent()

    def run(
        self,
        retrieval_result: RetrievalResult,
        ticker: str,
    ) -> SpecialistResults:
        print(f"\n{'='*50}")
        print(f"Running specialist agents in parallel for {ticker}...")
        start = time.time()

        results = {}

        # Define tasks
        tasks = {
            "financial": lambda: self.financial_agent.run(retrieval_result, ticker),
            "sentiment": lambda: self.sentiment_agent.run(retrieval_result, ticker),
            "risk":      lambda: self.risk_agent.run(retrieval_result, ticker),
        }

        # Run all three simultaneously
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fn): name
                for name, fn in tasks.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    print(f"  ✅ {name} agent completed")
                except Exception as e:
                    print(f"  ❌ {name} agent failed: {e}")
                    if name == "financial":
                        from agents.groq.financial_agent import FinancialExtractionResult
                        results[name] = FinancialExtractionResult(ticker=ticker, warnings=[str(e)])
                    elif name == "sentiment":
                        from agents.groq.sentiment_agent import SentimentResult
                        results[name] = SentimentResult(ticker=ticker, warnings=[str(e)])
                    elif name == "risk":
                        from agents.groq.risk_agent import RiskResult
                        results[name] = RiskResult(ticker=ticker, warnings=[str(e)])

        elapsed = round(time.time() - start, 2)
        print(f"\nAll specialist agents done in {elapsed}s")

        return SpecialistResults(
            financial=results["financial"],
            sentiment=results["sentiment"],
            risk=results["risk"],
            elapsed_seconds=elapsed,
        )


# Singleton
parallel_runner = ParallelAgentRunner()