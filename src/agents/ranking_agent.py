"""
Ranking Agent — uses DeepSeek-R1 to extract structured hotel data
from raw search results and rank them by value (price + rating).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from src.config.settings import get_settings


class RankingAgent:
    """
    Reads raw web search text and uses the LLM to extract
    structured hotel entries, then ranks them by score.
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.1)
        self._parser = JsonOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a strict hotel data extraction engine.
Extract ONLY hotels that are explicitly mentioned in the raw search text.

Rules:
- NEVER invent, guess, or assume hotel names, prices, or ratings
- If a hotel name is not clearly stated, skip it entirely
- If a price is not clearly stated, set price_per_night to null
- If a rating is not clearly stated, set rating to null
- Only include hotels in {city}, Egypt — ignore all other locations
- Do NOT include placeholders, generic names, or made-up entries
- If no real hotels are found, return: {{"hotels": []}}
- Output ONLY valid JSON, no explanation, no markdown, no code blocks

Required format:
{{
  "hotels": [
    {{
      "name": "Exact Hotel Name From Text",
      "city": "{city}",
      "price_per_night": 45.0,
      "rating": 4.2,
      "notes": "one sentence from the search text"
    }}
  ]
}}"""
            ),
            (
                "human",
                "City requested: {city}, Egypt\n"
                "Budget: ${budget} per night\n\n"
                "Raw search results:\n{raw_results}\n\n"
                "IMPORTANT: Only extract hotels located in {city}, Egypt. "
                "Ignore any results from other countries. "
                "Extract and return structured hotel data."
            ),
        ])

        self._chain = self._prompt | self._llm | self._parser

    def _score_hotel(self, hotel: dict, budget: float) -> float:
        """
        Computes a value score for one hotel.
        Higher score = better value for money.
        Formula: rating weight (70%) + budget savings weight (30%)
        """
        rating = hotel.get("rating") or 3.0
        price = hotel.get("price_per_night") or budget

        # How much of the budget is saved (0.0 to 1.0)
        budget_savings = max(0.0, (budget - price) / budget) if budget > 0 else 0.0

        score = (rating / 5.0) * 0.7 + budget_savings * 0.3
        return round(score, 4)

    def run(self, city: str, budget: float, raw_results: str) -> dict:
        """
        Extracts hotels from raw text and returns them ranked by value score.

        Returns a dict with:
          - ranked_hotels: list of hotel dicts sorted by score descending
          - total_found: how many hotels were extracted
        """
        print(f"[RankingAgent] Extracting hotels from raw results...")

        try:
            extracted = self._chain.invoke({
                "city": city,
                "budget": budget,
                "raw_results": raw_results,
            })
            hotels = extracted.get("hotels", [])
        except (OutputParserException, Exception) as e:
            print(f"[RankingAgent] Warning: extraction failed ({e}). Returning empty list.")
            hotels = []

        # Filter to only hotels within budget (price not null and within budget)
        within_budget = [
            h for h in hotels
            if h.get("price_per_night") is None or h["price_per_night"] <= budget
        ]

        # Score and sort
        for hotel in within_budget:
            hotel["score"] = self._score_hotel(hotel, budget)

        ranked = sorted(within_budget, key=lambda h: h["score"], reverse=True)

        print(f"[RankingAgent] {len(ranked)} hotels ranked within budget.")

        if not ranked:
            print("[RankingAgent] No hotels extracted. Raw results preview:")
            print(raw_results[:300])  # show first 300 chars for debugging

        return {
            "ranked_hotels": ranked,
            "total_found": len(hotels),
        }