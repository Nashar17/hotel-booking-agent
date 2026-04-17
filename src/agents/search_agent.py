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

    def _build_query(self, city: str, budget: float, attempt: int = 1) -> str:
        """
        Builds a search query. Second attempt uses a broader phrasing.
        """
        if attempt == 1:
            return f"hotels in {city} Egypt EGP price per night under {int(budget)} dollars"
        else:
            return f"best hotels {city} Alexandria Egypt Mediterranean coast affordable"
        
    def run(self, city: str, budget: float, check_in: str, check_out: str, attempt: int = 1) -> dict:
        query = self._build_query(city, budget, attempt)

        print(f"[SearchAgent] Query built: '{query}'")
        print("[SearchAgent] Fetching results from the web...")

        raw_results = self._tool._run(query)

        return {
            "query": query,
            "raw_results": raw_results,
        }