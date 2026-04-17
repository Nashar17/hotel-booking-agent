"""
Booking Graph — LangGraph state machine that orchestrates all agents.
This is the brain of the application. main.py calls only this.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from src.agents.search_agent import SearchAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.report_agent import ReportAgent


# ── State Definition ──────────────────────────────────────────────────────────
# This is the shared memory that flows through every node.
# Every key is written by exactly one node and read by the next.

class BookingState(TypedDict):
    # User inputs (set once at the start, never changed)
    city: str
    budget: float
    check_in: str
    check_out: str

    # Filled by SearchNode
    search_query: str
    raw_results: str

    # Filled by RankingNode
    ranked_hotels: List[dict]
    total_found: int

    # Filled by ReportNode
    final_report: str
    top_hotel: Optional[dict]

    # Control fields
    attempts: int
    error_message: Optional[str]


# ── Node Functions ────────────────────────────────────────────────────────────
# Each node is a plain function: receives full state, returns only changed keys.
# Agents are instantiated once inside each function call (stateless design).

def search_node(state: BookingState) -> dict:
    """Calls SearchAgent to fetch raw hotel data from the web."""
    agent = SearchAgent()
    result = agent.run(
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
    """Calls RankingAgent to extract and score hotels from raw results."""
    agent = RankingAgent()
    result = agent.run(
        city=state["city"],
        budget=state["budget"],
        raw_results=state["raw_results"],
    )
    return {
        "ranked_hotels": result["ranked_hotels"],
        "total_found": result["total_found"],
    }


def report_node(state: BookingState) -> dict:
    """Calls ReportAgent to produce the final human-readable recommendation."""
    agent = ReportAgent()
    result = agent.run(
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
    """Handles the case where max retries are exceeded with no results."""
    return {
        "final_report": (
            f"Sorry, I was unable to find hotels in {state['city']} "
            f"within your ${state['budget']}/night budget after "
            f"{state['attempts']} attempts. "
            "Please try a different city or increase your budget."
        ),
        "top_hotel": None,
    }


# ── Conditional Edge Functions ────────────────────────────────────────────────

def decide_after_ranking(state: BookingState) -> str:
    """
    After ranking, decide what to do next:
    - If hotels found → proceed to report
    - If no hotels and attempts < 2 → retry search
    - If no hotels and max attempts reached → go to error
    """
    has_results = len(state["ranked_hotels"]) > 0
    max_attempts_reached = state["attempts"] >= 2

    if has_results:
        return "proceed"
    elif not max_attempts_reached:
        return "retry"
    else:
        return "give_up"


# ── Graph Builder ─────────────────────────────────────────────────────────────

class BookingGraph:
    """
    Encapsulates the full LangGraph workflow.
    Exposes a single .run() method that main.py and the UI call.
    """

    def __init__(self):
        self._graph = self._build()

    def _build(self) -> object:
        """Constructs, wires, and compiles the StateGraph."""
        builder = StateGraph(BookingState)

        # Register all nodes
        builder.add_node("search", search_node)
        builder.add_node("rank", ranking_node)
        builder.add_node("report", report_node)
        builder.add_node("error", error_node)

        # Entry point
        builder.set_entry_point("search")

        # Fixed edges
        builder.add_edge("search", "rank")
        builder.add_edge("report", END)
        builder.add_edge("error", END)

        # Conditional edge after ranking
        builder.add_conditional_edges(
            "rank",
            decide_after_ranking,
            {
                "proceed": "report",
                "retry": "search",
                "give_up": "error",
            }
        )

        return builder.compile()

    def run(self, city: str, budget: float, check_in: str, check_out: str) -> dict:
        """
        Runs the full booking workflow.

        Returns a dict with:
          - final_report: string to display to the user
          - top_hotel: best hotel dict or None
          - ranked_hotels: full ranked list
          - attempts: how many search attempts were made
        """
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
            "error_message": None,
        }

        final_state = self._graph.invoke(initial_state)

        return {
            "final_report": final_state["final_report"],
            "top_hotel": final_state["top_hotel"],
            "ranked_hotels": final_state["ranked_hotels"],
            "attempts": final_state["attempts"],
        }