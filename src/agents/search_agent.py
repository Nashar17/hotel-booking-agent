"""
Search Agent — builds a hotel search query and fetches real hotel data.

Query construction is done in Python — reliable, fast, no hallucination risk.
Uses SerpApi Google Hotels for structured results with real prices worldwide.
"""

import logging
from src.tools.browser_tool import HotelSearchTool

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Builds a targeted hotel search and fetches real web results.

    Attempt 1: Exact city + budget via SerpApi Google Hotels.
    Attempt 2: Slightly relaxed budget (+20%) to surface more options
               when the first attempt returns nothing.
    """

    def __init__(self):
        self._tool = HotelSearchTool()

    def _get_relaxed_budget(self, budget: float) -> float:
        """
        On retry, expand the budget ceiling by 20% to surface
        more results that might still be close to what the user wants.
        The RankingAgent will still filter to the original budget.
        """
        return round(budget * 1.2, 2)

    def run(
        self,
        city: str,
        budget: float,
        check_in: str,
        check_out: str,
        attempt: int = 1,
    ) -> dict:
        """
        Fetches hotel results for the given parameters.

        On attempt 2, the budget ceiling is relaxed by 20% so SerpApi
        returns more candidates — the RankingAgent still enforces the
        original budget in its filter step.

        Returns:
            {
                "query": str,         — human-readable description of what was searched
                "raw_results": str,   — formatted text for RankingAgent to parse
            }
        """
        effective_budget = budget if attempt == 1 else self._get_relaxed_budget(budget)

        query_description = (
            f"hotels in {city}, budget up to ${effective_budget}/night, "
            f"{check_in} → {check_out}"
            + (" [retry with relaxed budget]" if attempt > 1 else "")
        )

        logger.info(
            "SearchAgent attempt %d: %s",
            attempt,
            query_description,
        )

        raw_results = self._tool._run(
            city=city,
            budget=effective_budget,
            check_in=check_in,
            check_out=check_out,
        )

        logger.debug("Raw results preview: %s", raw_results[:200])

        return {
            "query": query_description,
            "raw_results": raw_results,
        }