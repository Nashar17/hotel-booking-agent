"""
Report Agent — takes ranked hotel data and produces a clean,
human-readable final response for the Streamlit UI.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_settings
import json


class ReportAgent:
    """
    Formats ranked hotel data into a polished, human-friendly report.
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.4)
        self._parser = StrOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a hotel recommendation assistant. "
                "Write ONLY based on the hotel data provided to you. "
                "Do NOT add any facts, landmarks, attractions, or details "
                "that are not present in the hotel data. "
                "If data is limited, say so honestly rather than inventing details. "
                "Keep the response under 120 words. Warm, helpful tone. "
                "When mentioning prices always use the $ symbol directly, never backticks. "
                "Example: $150/night not `150/night`."
            ),
            (
                "human",
                "City: {city}, Egypt\n"
                "Check-in: {check_in} | Check-out: {check_out}\n"
                "Budget: ${budget}/night\n\n"
                "Ranked hotels found:\n{hotels_json}\n\n"
                "Write a short, friendly recommendation report."
                "Make clear these hotels are in {city}, Egypt."
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

        Returns a dict with:
          - report: the formatted string to show the user
          - top_hotel: the single best hotel dict (for UI display)
        """
        print("[ReportAgent] Generating final recommendation...")

        if not ranked_hotels:
            return {
                "report": f"Sorry, I couldn't find any hotels in {city} within your ${budget}/night budget. Try increasing your budget or searching a nearby city.",
                "top_hotel": None,
            }

        # Only pass top 5 to keep the prompt focused
        top_hotels = ranked_hotels[:5]
        hotels_json = json.dumps(top_hotels, indent=2)

        report = self._chain.invoke({
            "city": city,
            "budget": budget,
            "check_in": check_in,
            "check_out": check_out,
            "hotels_json": hotels_json,
        })

        return {
            "report": report,
            "top_hotel": ranked_hotels[0] if ranked_hotels else None,
        }