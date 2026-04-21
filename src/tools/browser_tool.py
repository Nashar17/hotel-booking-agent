"""
Browser Tool — fetches real hotel data using the SerpApi Google Hotels API.

Why SerpApi instead of DuckDuckGo?
  - Returns structured hotel data (name, price, rating, reviews) directly
  - Supports worldwide cities with check-in/check-out date filtering
  - No HTML scraping or LLM extraction needed for basic fields
  - Free tier: 100 searches/month — enough for demos and interviews

Docs: https://serpapi.com/google-hotels-api
"""

import logging
import requests
from typing import Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class HotelSearchInput(BaseModel):
    """
    Input schema for the HotelSearchTool.
    Pydantic validates that the caller passes the correct argument types.
    """
    city: str = Field(description="City to search hotels in. Example: 'Paris', 'Tokyo', 'Cairo'")
    budget: float = Field(description="Maximum budget per night in USD")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format")


class HotelSearchTool(BaseTool):
    """
    A LangChain tool that queries the SerpApi Google Hotels API for real
    hotel listings with prices, ratings, and availability worldwide.

    Falls back gracefully if the API key is missing or the request fails.
    """

    name: str = "hotel_search"
    description: str = (
        "Search for hotels in any city worldwide with real prices and ratings. "
        "Provide city name, maximum budget per night in USD, and travel dates."
    )
    args_schema: type[BaseModel] = HotelSearchInput

    def _run(
        self,
        city: str,
        budget: float,
        check_in: str,
        check_out: str,
    ) -> str:
        """
        Queries SerpApi Google Hotels and returns structured results as text.
        The RankingAgent reads this text and extracts hotel records from it.
        """
        settings = get_settings()

        if not settings.serpapi_key:
            logger.warning("SERPAPI_KEY not set — falling back to DuckDuckGo")
            return self._duckduckgo_fallback(city, budget)

        logger.info("Querying SerpApi Google Hotels: city=%s, budget=$%s", city, budget)

        params = {
            "engine": "google_hotels",
            "q": f"hotels in {city}",
            "check_in_date": check_in,
            "check_out_date": check_out,
            "adults": "2",
            "currency": "USD",
            "sort_by": 3,            # 3 = sort by lowest price (must be int)
            "gl": "us",              # Use US locale for USD prices
            "hl": "en",
            "api_key": settings.serpapi_key,
        }
        # Note: max_price is not a valid SerpApi Google Hotels parameter.
        # Budget filtering is handled downstream by RankingAgent.

        try:
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=settings.search_timeout,
            )
            response.raise_for_status()
            data = response.json()

            hotels = data.get("properties", [])

            if not hotels:
                logger.warning("SerpApi returned 0 hotels for city=%s", city)
                return f"No hotels found in {city} within ${budget}/night budget via Google Hotels."

            logger.info("SerpApi returned %d hotel(s) for %s", len(hotels), city)
            return self._format_results(hotels, city, budget)

        except requests.exceptions.Timeout:
            logger.error("SerpApi request timed out for city=%s", city)
            return f"Search timed out for {city}. Please try again."

        except requests.exceptions.RequestException as e:
            logger.error("SerpApi request failed: %s", e)
            return self._duckduckgo_fallback(city, budget)

    def _format_results(self, hotels: list, city: str, budget: float) -> str:
        """
        Converts SerpApi hotel objects into a clean text block that
        RankingAgent can easily parse. Each hotel is one clearly
        delimited section.
        """
        lines = [f"Google Hotels results for {city} (budget: up to ${budget}/night):\n"]

        for i, hotel in enumerate(hotels[:10], 1):
            name = hotel.get("name", "Unknown Hotel")
            
            # Price — SerpApi returns rate_per_night as a dict with 'lowest'
            rate = hotel.get("rate_per_night", {})
            price_raw = rate.get("lowest", "")
            price_str = price_raw if price_raw else "Price not available"

            # Rating
            rating = hotel.get("overall_rating", "N/A")
            reviews = hotel.get("reviews", "")
            
            # Description / highlights
            description = hotel.get("description", "")
            amenities = hotel.get("amenities", [])
            amenity_str = ", ".join(amenities[:5]) if amenities else ""

            lines.append(f"--- Hotel {i} ---")
            lines.append(f"Name: {name}")
            lines.append(f"Location: {city}")
            lines.append(f"Price per night: {price_str}")
            lines.append(f"Rating: {rating}/5 ({reviews} reviews)")
            if description:
                lines.append(f"Description: {description[:200]}")
            if amenity_str:
                lines.append(f"Amenities: {amenity_str}")
            lines.append("")

        return "\n".join(lines)

    def _duckduckgo_fallback(self, city: str, budget: float) -> str:
        """
        Fallback to DuckDuckGo if SerpApi is unavailable.
        Less reliable for prices but better than nothing.
        """
        logger.info("Using DuckDuckGo fallback for city=%s", city)
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            query = f"best hotels in {city} under {int(budget)} USD per night price rating"
            result = search.run(query)
            return f"Web search results for hotels in {city}:\n{result}"
        except Exception as e:
            logger.error("DuckDuckGo fallback also failed: %s", e)
            return f"Search unavailable. Could not find hotels in {city} at this time."

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("HotelSearchTool does not support async.")