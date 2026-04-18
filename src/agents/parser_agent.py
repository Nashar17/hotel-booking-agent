"""
Parser Agent — extracts structured hotel search parameters
from a free-form natural language user message.
Returns what was found and what information is still missing.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from src.config.settings import get_settings
from datetime import date, timedelta


class ParserAgent:
    """
    Reads the full conversation history and extracts:
    - city (required)
    - budget in USD per night (required)
    - check_in date (required)
    - check_out date (required)

    Returns a dict with extracted values and a list of what's still missing.
    """

    def __init__(self):
        settings = get_settings()
        self._llm = settings.build_llm(temperature=0.0)
        self._parser = JsonOutputParser()

        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a hotel booking assistant that extracts search parameters
from a conversation. Analyze the full conversation and extract any hotel search info.

Today's date is {today}.

Extract and return ONLY this JSON — no explanation, no markdown:
{{
  "city": "city name or null if not mentioned",
  "budget": number in USD or null if not mentioned,
  "check_in": "YYYY-MM-DD or null if not mentioned",
  "check_out": "YYYY-MM-DD or null if not mentioned",
  "missing": ["list of fields still needed"],
  "clarification_question": "one friendly question asking for ALL missing fields at once, or null if nothing is missing"
}}

Rules:
- If user says "3 nights from August 1st" → check_in: that date, check_out: 3 days later
- If user says "this weekend" → calculate from today's date
- If user says "under $200" or "200 dollars" → budget: 200
- If user says "best 3 hotels" → that means budget is still unknown, add to missing
- missing list must contain only the fields that are truly null
- If nothing is missing, clarification_question must be null"""
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

        conversation_history: list of dicts like:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            {
                "city": str or None,
                "budget": float or None,
                "check_in": str or None,
                "check_out": str or None,
                "missing": list,
                "clarification_question": str or None,
                "ready_to_search": bool
            }
        """
        # Format conversation as readable text for the LLM
        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation_history
        ])

        today = date.today().strftime("%Y-%m-%d")

        try:
            result = self._chain.invoke({
                "conversation": conversation_text,
                "today": today,
            })

            result["ready_to_search"] = len(result.get("missing", [])) == 0
            return result

        except (OutputParserException, Exception) as e:
            print(f"[ParserAgent] Extraction failed: {e}")
            return {
                "city": None,
                "budget": None,
                "check_in": None,
                "check_out": None,
                "missing": ["city", "budget", "check_in", "check_out"],
                "clarification_question": "I'd love to help! Could you tell me which city, your budget per night, and your check-in and check-out dates?",
                "ready_to_search": False,
            }