"""
Report Agent — takes ranked hotel data and produces a clean,
human-readable final recommendation for the Streamlit UI.
Works for any city worldwide.
"""

import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Formats ranked hotel data into a polished, human-friendly report.
    Only uses facts present in the hotel data — never invents details.
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.4)
        self._parser = StrOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a friendly hotel recommendation assistant. "
                "Write ONLY based on the hotel data provided — never invent landmarks, "
                "attractions, or details not present in the data. "
                "If price data is limited, say so honestly rather than guessing. "
                "Keep the response under 200 words. Warm, helpful, conversational tone. "
                "CRITICAL FORMATTING RULE: Always write prices like this: $150/night — "
                "NEVER use backticks around prices. NEVER write `150/night` or `$150`. "
                "The dollar sign must appear directly before the number with no backticks anywhere. "
                "Mention the city name naturally in the response."
            ),
            (
                "human",
                "City: {city}\n"
                "Check-in: {check_in} | Check-out: {check_out}\n"
                "Budget: ${budget}/night\n\n"
                "Ranked hotels found:\n{hotels_json}\n\n"
                "Write a short, friendly recommendation. "
                "Highlight the top pick and briefly mention the runners-up. "
                "If prices are missing for some hotels, note that honestly."
            ),
        ])

        self._chain = self._prompt | self._llm | self._parser

    def run(
        self,
        city: str,
        budget: float,
        check_in: str,
        check_out: str,
        ranked_hotels: list,
    ) -> dict:
        """
        Generates the final human-readable report.

        Returns:
            {
                "report": str,          — formatted recommendation string
                "top_hotel": dict|None  — best hotel dict for UI display
            }
        """
        logger.info(
            "ReportAgent generating recommendation for %d hotel(s) in %s",
            len(ranked_hotels),
            city,
        )

        if not ranked_hotels:
            logger.warning("No hotels to report for city=%s", city)
            return {
                "report": (
                    f"Sorry, I couldn't find any hotels in **{city}** within your "
                    f"**${budget}/night** budget. You could try:\n"
                    f"- Increasing your budget slightly\n"
                    f"- Searching a nearby major city\n"
                    f"- Adjusting your travel dates"
                ),
                "top_hotel": None,
            }

        # Pass top 5 to keep the prompt focused and tokens low
        top_hotels = ranked_hotels[:5]
        hotels_json = json.dumps(top_hotels, indent=2)

        report = self._chain.invoke({
            "city": city,
            "budget": budget,
            "check_in": check_in,
            "check_out": check_out,
            "hotels_json": hotels_json,
        })

        logger.info("ReportAgent finished — top hotel: %s", ranked_hotels[0].get("name"))

        return {
            "report": report,
            "top_hotel": ranked_hotels[0],
        }