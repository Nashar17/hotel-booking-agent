"""
Ranking Agent — extracts structured hotel data from raw search results
and ranks them by value (price + rating score).

Works for any city worldwide — no country hardcoding.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class RankingAgent:
    """
    Reads raw web/API search text and uses the LLM to extract
    structured hotel entries, then ranks them by value score.

    Scoring formula:
        score = (rating / 5.0) * 0.7  +  (budget_savings / budget) * 0.3
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.1)
        self._parser = JsonOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a strict hotel data extraction engine.
Extract ONLY hotels that are explicitly located in {city}.

Rules:
- Extract hotels ONLY from the city of {city} in the search results
- NEVER invent, guess, or assume hotel names, prices, or ratings
- If a hotel has no name or an empty name, skip it entirely
- Prices should be numbers only (no currency symbols)
- If a price appears as "$120" extract it as 120.0
- If no hotels are found, return: {{"hotels": []}}
- Output ONLY valid JSON, no explanation, no markdown, no code blocks

Required format:
{{
  "hotels": [
    {{
      "name": "Exact Hotel Name From Text",
      "city": "{city}",
      "price_per_night": 120.0,
      "rating": 4.2,
      "notes": "one short sentence summarizing the hotel from the search text"
    }}
  ]
}}"""
            ),
            (
                "human",
                "City requested: {city}\n"
                "Budget: ${budget} per night\n\n"
                "Raw search results:\n{raw_results}\n\n"
                "Extract and return structured hotel data for hotels in {city} only."
            ),
        ])

        self._chain = self._prompt | self._llm | self._parser

    def _is_valid_hotel(self, hotel: dict) -> bool:
        """
        Returns True only if the hotel record has a non-empty name.
        Filters out any hallucinated or malformed entries.
        """
        name = hotel.get("name", "")
        if not name or not isinstance(name, str) or name.strip() == "":
            logger.warning("Skipping hotel with missing/empty name: %s", hotel)
            return False
        return True

    def _score_hotel(self, hotel: dict, budget: float) -> float:
        """
        Computes a value score for one hotel.
        Higher score = better value for money.

        Formula:
            70% weight → rating (higher is better)
            30% weight → how much budget is saved (lower price = higher score)
        """
        rating = hotel.get("rating") or 3.0
        price = hotel.get("price_per_night") or budget

        budget_savings = max(0.0, (budget - price) / budget) if budget > 0 else 0.0
        score = (rating / 5.0) * 0.7 + budget_savings * 0.3

        return round(score, 4)

    def run(self, city: str, budget: float, raw_results: str) -> dict:
        """
        Extracts hotels from raw text and returns them ranked by value score.

        Returns:
            {
                "ranked_hotels": list[dict],   — sorted by score descending
                "total_found": int,            — total extracted before filtering
            }
        """
        logger.info("RankingAgent extracting hotels for city=%s, budget=$%s", city, budget)

        try:
            extracted = self._chain.invoke({
                "city": city,
                "budget": budget,
                "raw_results": raw_results,
            })
            hotels = extracted.get("hotels", [])
            logger.info("LLM extracted %d hotel(s) before validation", len(hotels))

        except (OutputParserException, Exception) as e:
            logger.error("RankingAgent extraction failed: %s", e)
            hotels = []

        # Validate: must have a real name
        valid_hotels = [h for h in hotels if self._is_valid_hotel(h)]

        # Filter: within budget (allow nulls through — ReportAgent handles them honestly)
        within_budget = [
            h for h in valid_hotels
            if h.get("price_per_night") is None or h["price_per_night"] <= budget
        ]

        # Score and sort
        for hotel in within_budget:
            hotel["score"] = self._score_hotel(hotel, budget)

        ranked = sorted(within_budget, key=lambda h: h["score"], reverse=True)

        logger.info(
            "RankingAgent: %d valid hotel(s) within budget after filtering",
            len(ranked),
        )

        if not ranked:
            logger.warning(
                "No hotels passed filters. Raw results preview:\n%s",
                raw_results[:400],
            )

        return {
            "ranked_hotels": ranked,
            "total_found": len(hotels),
        }