"""
Parser Agent — extracts structured hotel search parameters
from a free-form natural language conversation.
Works for any city worldwide — no country assumptions.
"""

import logging
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ParserAgent:
    """
    Reads the full conversation history and extracts:
      - city        (required) — any city worldwide
      - budget      (required) — max USD per night
      - check_in    (required) — YYYY-MM-DD
      - check_out   (required) — YYYY-MM-DD

    Returns a dict with extracted values, missing fields list,
    an optional clarification question, and a ready_to_search flag.
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.0)
        self._parser = JsonOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a hotel booking assistant that extracts search parameters
from a conversation. Analyze the full conversation and extract hotel search info.

Today's date is {today}.

Extract and return ONLY this JSON — no explanation, no markdown:
{{
  "city": "city name or null if not mentioned",
  "budget": number in USD per night or null if not mentioned,
  "check_in": "YYYY-MM-DD or null if not mentioned",
  "check_out": "YYYY-MM-DD or null if not mentioned",
  "missing": ["list of field names still needed"],
  "clarification_question": "one friendly question asking for ALL missing fields at once, or null if nothing is missing"
}}

Rules:
- city can be ANY city worldwide — do not assume a country
- If user says "3 nights from August 1st" → check_in: that date, check_out: 3 days later
- If user says "this weekend" → calculate from today's date
- If user says "under $200" or "200 dollars" → budget: 200
- If user says "best hotels" with no budget → budget is still missing, add to missing list
- missing list must contain only the fields that are truly null
- If nothing is missing, clarification_question must be null
- Ask for only one clarification at a time if multiple fields are missing — ask for all of them in one friendly question"""
            ),
            (
                "human",
                "Conversation so far:\n{conversation}\n\nExtract the search parameters."
            ),
        ])

        self._chain = self._prompt | self._llm | self._parser

    def run(self, conversation_history: list[dict]) -> dict:
        """
        Analyzes conversation history and returns extracted parameters.

        Args:
            conversation_history: list of dicts like:
                [{"role": "user", "content": "..."}, ...]

        Returns:
            {
                "city": str | None,
                "budget": float | None,
                "check_in": str | None,
                "check_out": str | None,
                "missing": list[str],
                "clarification_question": str | None,
                "ready_to_search": bool,
            }
        """
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation_history
        )

        today = date.today().strftime("%Y-%m-%d")

        logger.debug("ParserAgent analyzing conversation (%d messages)", len(conversation_history))

        try:
            result = self._chain.invoke({
                "conversation": conversation_text,
                "today": today,
            })

            result["ready_to_search"] = len(result.get("missing", [])) == 0

            logger.info(
                "ParserAgent extracted: city=%s, budget=%s, check_in=%s, check_out=%s | missing=%s",
                result.get("city"),
                result.get("budget"),
                result.get("check_in"),
                result.get("check_out"),
                result.get("missing"),
            )

            return result

        except (OutputParserException, Exception) as e:
            logger.error("ParserAgent extraction failed: %s", e)
            return {
                "city": None,
                "budget": None,
                "check_in": None,
                "check_out": None,
                "missing": ["city", "budget", "check_in", "check_out"],
                "clarification_question": (
                    "I'd love to help! Could you tell me:\n"
                    "- Which city are you travelling to?\n"
                    "- Your budget per night (in USD)?\n"
                    "- Your check-in and check-out dates?"
                ),
                "ready_to_search": False,
            }