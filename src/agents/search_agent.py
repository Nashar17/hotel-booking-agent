"""
Search Agent — responsible for querying the web tool to find raw hotel data.
Builds the search query directly in Python (no LLM needed for this step).
"""

from src.tools.browser_tool import HotelSearchTool


class SearchAgent:
    """
    Builds a targeted hotel search query and fetches real web results.
    Query construction is done in Python — reliable, fast, no hallucination risk.
    """

    def __init__(self):
        self._tool = HotelSearchTool()

    def _build_query(self, city: str, budget: float) -> str:
        """
        Builds a clean, effective search query from structured inputs.
        No LLM involved — Python string formatting is more reliable here.
        """
        return f"hotels in {city} Egypt budget under {int(budget)} USD per night"

    def run(self, city: str, budget: float, check_in: str, check_out: str) -> dict:
        """
        Builds an optimized search query and fetches real hotel results.

        Returns a dict with:
          - query: the search string used
          - raw_results: the raw text from DuckDuckGo
        """
        query = self._build_query(city, budget)

        print(f"[SearchAgent] Query built: '{query}'")
        print("[SearchAgent] Fetching results from the web...")

        raw_results = self._tool._run(query)

        return {
            "query": query,
            "raw_results": raw_results,
        }