"""
Test the Query Understanding Agent with different query types.
Run: python test_query_agent.py
"""

import json
from agents.query_agent_groq import query_agent

TEST_QUERIES = [
    # Beginner
    "What is gross margin and why does it matter?",
    # Simple research
    "What was Nvidia's revenue in Q3 2024?",
    # Trend analysis
    "How has Nvidia's pricing power changed over the last 4 quarters?",
    # Comparative
    "Compare Apple and Microsoft cloud revenue growth in 2024",
    # Risk focused
    "What risks did Nvidia management mention about China exports?",
    # Real-time
    "What is Nvidia's current stock price and PE ratio?",
    # Complex multi-part
    "How did Nvidia's gross margin and guidance change from Q1 to Q4 2024 and what did analysts ask about?",
]


def print_plan(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    plan = query_agent.run(query)

    print(f"Type:          {plan.question_type}")
    print(f"Complexity:    {plan.complexity}")
    print(f"User level:    {plan.user_level}")
    print(f"Response fmt:  {plan.response_format}")
    print(f"Tickers:       {plan.entities.tickers}")
    print(f"Quarters:      {plan.entities.quarters}")
    print(f"Years:         {plan.entities.years}")
    print(f"Metrics:       {plan.entities.metrics}")
    print(f"Time range:    {plan.entities.time_range}")
    print(f"\nSub-questions ({len(plan.sub_questions)}):")
    for i, sq in enumerate(plan.sub_questions, 1):
        print(f"  {i}. {sq.question}")
        if sq.ticker:   print(f"     ticker={sq.ticker}")
        if sq.metric:   print(f"     metric={sq.metric}")
        if sq.section:  print(f"     section={sq.section}")

    rp = plan.retrieval_plan
    print(f"\nRetrieval plan:")
    print(f"  Sources:     {rp.sources_needed}")
    print(f"  Sections:    {rp.sections}")
    print(f"  N results:   {rp.n_results}")
    print(f"  Comparison:  {rp.requires_comparison}")
    print(f"  Calculation: {rp.requires_calculation}")


if __name__ == "__main__":
    for query in TEST_QUERIES:
        print_plan(query)