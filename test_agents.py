"""
Integration test for all three agents.
Run with: python test_agents.py
"""

from src.agents.search_agent import SearchAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.report_agent import ReportAgent

# ── Inputs ────────────────────────────────────────────────────────────────────
city = "Alexandria"
budget = 60.0
check_in = "2025-08-01"
check_out = "2025-08-04"

# ── Run each agent in sequence ────────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 1: Search Agent")
print("="*55)
search_agent = SearchAgent()
search_output = search_agent.run(city, budget, check_in, check_out)
print(f"Query used: {search_output['query']}")

print("\n" + "="*55)
print("STEP 2: Ranking Agent")
print("="*55)
ranking_agent = RankingAgent()
ranking_output = ranking_agent.run(city, budget, search_output["raw_results"])
print(f"Hotels found: {ranking_output['total_found']}")
print(f"Within budget: {len(ranking_output['ranked_hotels'])}")
for h in ranking_output["ranked_hotels"]:
    print(f"  - {h['name']} | ${h.get('price_per_night')} | score: {h.get('score')}")

print("\n" + "="*55)
print("STEP 3: Report Agent")
print("="*55)
report_agent = ReportAgent()
report_output = report_agent.run(
    city, budget, check_in, check_out,
    ranking_output["ranked_hotels"]
)
print("\nFINAL REPORT:")
print(report_output["report"])