"""
Pipeline Orchestrator
----------------------
Ties all agents together into a single end-to-end pipeline.
This is the main entry point for answering investor queries.

Flow:
  Query → QueryAgent → RetrievalAgent → ParallelSpecialists → SynthesisAgent → Response
"""

import time
from agents.gemini.query_agent import query_agent, QueryPlan
from agents.gemini.retrieval_agent import retrieval_agent, RetrievalResult
from agents.gemini.parallel_runner import parallel_runner, SpecialistResults
from agents.gemini.synthesis_agent import synthesis_agent, SynthesisResult


class EarningsAnalyzerPipeline:

    def run(self, query: str, verbose: bool = True) -> SynthesisResult:
        start = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}")

        # ── Step 1: Query Understanding ────────────────────────────────
        plan: QueryPlan = query_agent.run(query)
        if verbose:
            print(f"Plan: type={plan.question_type} | "
                  f"tickers={plan.entities.tickers} | "
                  f"complexity={plan.complexity} | "
                  f"user_level={plan.user_level}")

        # ── Step 2: Retrieval ──────────────────────────────────────────
        retrieval: RetrievalResult = retrieval_agent.run(plan)
        if verbose:
            print(f"Retrieved: {retrieval.total_found} chunks | "
                  f"Quality: {retrieval.retrieval_quality} | "
                  f"Sources: {retrieval.sources_used}")

        # ── Step 3: Specialist Agents (parallel) ───────────────────────
        ticker = plan.entities.tickers[0] if plan.entities.tickers else "UNKNOWN"
        specialist: SpecialistResults = parallel_runner.run(retrieval, ticker)

        # ── Step 4: Synthesis ──────────────────────────────────────────
        result: SynthesisResult = synthesis_agent.run(plan, retrieval, specialist)

        elapsed = round(time.time() - start, 2)

        if verbose:
            print(f"\n{'='*60}")
            print(f"ANSWER (confidence={result.confidence} | "
                  f"quality={result.data_quality} | {elapsed}s total)")
            print(f"{'='*60}")
            print(result.answer)

            if result.references:
                print(f"\n📄 REFERENCES ({len(result.references)})")
                for i, ref in enumerate(result.references, 1):
                    print(f"  [{i}] {ref.title}")
                    if ref.source_url:
                        print(f"       {ref.source_url}")
                    if ref.relevant_excerpt:
                        print(f"       \"{ref.relevant_excerpt[:100]}...\"")

            if result.warnings:
                print(f"\n⚠️  WARNINGS")
                for w in result.warnings:
                    print(f"  • {w}")

        return result


# Singleton
pipeline = EarningsAnalyzerPipeline()