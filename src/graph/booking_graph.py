"""
Booking Graph — LangGraph state machine that orchestrates all agents.
Agents are instantiated once in BookingGraph.__init__ and reused
across all node calls — no unnecessary re-creation on every run.
"""

import logging
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from src.agents.search_agent import SearchAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


# ── State Definition ──────────────────────────────────────────────────────────
# Shared memory that flows through every node.
# Each key is written by exactly one node and read by the next.

class BookingState(TypedDict):
    # User inputs — set once at the start, never changed
    city: str
    budget: float
    check_in: str
    check_out: str

    # Filled by search_node
    search_query: str
    raw_results: str

    # Filled by ranking_node
    ranked_hotels: List[dict]
    total_found: int

    # Filled by report_node or error_node
    final_report: str
    top_hotel: Optional[dict]

    # Control — tracks retry attempts
    attempts: int


# ── Graph Builder ─────────────────────────────────────────────────────────────

class BookingGraph:
    """
    Encapsulates the full LangGraph workflow.
    Agents are created once and injected into node closures.
    Exposes a single .run() method called by the UI and CLI.
    """

    def __init__(self):
        # Instantiate agents once — reused across all graph executions
        self._search_agent = SearchAgent()
        self._ranking_agent = RankingAgent()
        self._report_agent = ReportAgent()

        self._graph = self._build()
        logger.info("BookingGraph initialized and compiled.")

    def _build(self) -> object:
        """Constructs, wires, and compiles the StateGraph."""

        # ── Node functions (closures over self._*_agent) ──────────────────────

        def search_node(state: BookingState) -> dict:
            """Fetches raw hotel data from the web."""
            logger.info("Graph → search_node (attempt %d)", state["attempts"] + 1)
            result = self._search_agent.run(
                city=state["city"],
                budget=state["budget"],
                check_in=state["check_in"],
                check_out=state["check_out"],
                attempt=state["attempts"] + 1,
            )
            return {
                "search_query": result["query"],
                "raw_results": result["raw_results"],
                "attempts": state["attempts"] + 1,
            }

        def ranking_node(state: BookingState) -> dict:
            """Extracts and scores hotels from raw results."""
            logger.info("Graph → ranking_node")
            result = self._ranking_agent.run(
                city=state["city"],
                budget=state["budget"],
                raw_results=state["raw_results"],
            )
            return {
                "ranked_hotels": result["ranked_hotels"],
                "total_found": result["total_found"],
            }

        def report_node(state: BookingState) -> dict:
            """Generates the final human-readable recommendation."""
            logger.info("Graph → report_node")
            result = self._report_agent.run(
                city=state["city"],
                budget=state["budget"],
                check_in=state["check_in"],
                check_out=state["check_out"],
                ranked_hotels=state["ranked_hotels"],
            )
            return {
                "final_report": result["report"],
                "top_hotel": result["top_hotel"],
            }

        def error_node(state: BookingState) -> dict:
            """Handles max retries exceeded with no results."""
            logger.warning(
                "Graph → error_node after %d attempt(s) for city=%s",
                state["attempts"],
                state["city"],
            )
            return {
                "final_report": (
                    f"Sorry, I was unable to find hotels in **{state['city']}** "
                    f"within your **${state['budget']}/night** budget after "
                    f"{state['attempts']} search attempt(s).\n\n"
                    "Suggestions:\n"
                    "- Try a slightly higher budget\n"
                    "- Search for a nearby major city\n"
                    "- Adjust your travel dates"
                ),
                "top_hotel": None,
            }

        # ── Conditional edge ──────────────────────────────────────────────────

        def decide_after_ranking(state: BookingState) -> str:
            """
            After ranking, decide the next step:
              - Hotels found       → proceed to report
              - No hotels, < 2 attempts → retry search with relaxed budget
              - No hotels, max attempts → give up gracefully
            """
            has_results = len(state["ranked_hotels"]) > 0
            max_attempts_reached = state["attempts"] >= 2

            if has_results:
                logger.debug("decide_after_ranking → proceed")
                return "proceed"
            elif not max_attempts_reached:
                logger.debug("decide_after_ranking → retry")
                return "retry"
            else:
                logger.debug("decide_after_ranking → give_up")
                return "give_up"

        # ── Assemble the graph ────────────────────────────────────────────────

        builder = StateGraph(BookingState)

        builder.add_node("search", search_node)
        builder.add_node("rank", ranking_node)
        builder.add_node("report", report_node)
        builder.add_node("error", error_node)

        builder.set_entry_point("search")

        builder.add_edge("search", "rank")
        builder.add_edge("report", END)
        builder.add_edge("error", END)

        builder.add_conditional_edges(
            "rank",
            decide_after_ranking,
            {
                "proceed": "report",
                "retry": "search",
                "give_up": "error",
            },
        )

        return builder.compile()

    def run(self, city: str, budget: float, check_in: str, check_out: str) -> dict:
        """
        Runs the full booking workflow end-to-end.

        Returns:
            {
                "final_report": str,          — text to display in UI
                "top_hotel":    dict | None,  — best hotel or None
                "ranked_hotels": list[dict],  — full sorted list
                "attempts":     int,          — how many search attempts were made
            }
        """
        logger.info(
            "BookingGraph.run: city=%s, budget=$%s, %s → %s",
            city, budget, check_in, check_out,
        )

        initial_state: BookingState = {
            "city": city,
            "budget": budget,
            "check_in": check_in,
            "check_out": check_out,
            "search_query": "",
            "raw_results": "",
            "ranked_hotels": [],
            "total_found": 0,
            "final_report": "",
            "top_hotel": None,
            "attempts": 0,
        }

        final_state = self._graph.invoke(initial_state)

        logger.info(
            "BookingGraph complete: %d hotel(s), %d attempt(s)",
            len(final_state["ranked_hotels"]),
            final_state["attempts"],
        )

        return {
            "final_report": final_state["final_report"],
            "top_hotel": final_state["top_hotel"],
            "ranked_hotels": final_state["ranked_hotels"],
            "attempts": final_state["attempts"],
        }