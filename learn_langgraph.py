"""
LangGraph Learning File — Phase 2B
This file is for learning only.
Run it with: python learn_langgraph.py
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END


# ── Step 1: Define the State ──────────────────────────────────────────────────
# State is a TypedDict — a dictionary with fixed, typed keys.
# Every node in the graph reads from and writes to this shared state.
# Think of it as the "memory" passed between agents.

class HotelSearchState(TypedDict):
    city: str                    # input: where the user wants to stay
    budget: float                # input: max price per night
    search_results: List[str]    # filled by: search node
    ranked_results: List[str]    # filled by: ranking node
    final_report: str            # filled by: report node
    attempts: int                # tracks how many search attempts we've made


# ── Step 2: Define the Nodes ──────────────────────────────────────────────────
# Each node is a plain Python function.
# It receives the full state dict, does its work, and returns ONLY the keys it changed.
# LangGraph merges the returned dict back into the full state automatically.

def search_node(state: HotelSearchState) -> dict:
    """Simulates searching for hotels. In the real project, this calls a web tool."""
    print(f"\n[Search Node] Searching hotels in {state['city']} under ${state['budget']}/night...")

    # Simulated results (later this will be real scraped data)
    fake_results = [
        f"Hotel Sunrise - {state['city']} - $45/night - Rating: 4.1",
        f"Palace Inn - {state['city']} - $55/night - Rating: 3.8",
        f"Budget Stay - {state['city']} - $30/night - Rating: 3.5",
        f"Grand View - {state['city']} - $80/night - Rating: 4.7",
    ]

    return {
        "search_results": fake_results,
        "attempts": state["attempts"] + 1
    }


def ranking_node(state: HotelSearchState) -> dict:
    """Filters results by budget and sorts by rating."""
    print(f"\n[Ranking Node] Filtering and ranking {len(state['search_results'])} results...")

    # Filter to only hotels within budget
    within_budget = []
    for hotel in state["search_results"]:
        # Extract price from the string (simplified parsing)
        price_part = hotel.split("$")[1].split("/")[0]
        price = float(price_part)
        if price <= state["budget"]:
            within_budget.append(hotel)

    # Sort by rating (last number in the string)
    ranked = sorted(within_budget, key=lambda h: float(h.split("Rating: ")[1]), reverse=True)

    print(f"[Ranking Node] {len(ranked)} hotels within budget after filtering.")

    return {"ranked_results": ranked}


def report_node(state: HotelSearchState) -> dict:
    """Formats the final answer to show the user."""
    print(f"\n[Report Node] Generating final report...")

    if not state["ranked_results"]:
        report = f"Sorry, no hotels found in {state['city']} under ${state['budget']}/night."
    else:
        lines = [f"Top hotels in {state['city']} under ${state['budget']}/night:\n"]
        for i, hotel in enumerate(state["ranked_results"], 1):
            lines.append(f"  {i}. {hotel}")
        report = "\n".join(lines)

    return {"final_report": report}


# ── Step 3: Conditional edge function ────────────────────────────────────────
# This function decides which node to go to NEXT based on the current state.
# It returns a string that matches one of the edge destinations.

def should_retry(state: HotelSearchState) -> str:
    """
    After ranking, check if we got results.
    If not, retry search (max 2 attempts). Otherwise, proceed to report.
    """
    if len(state["ranked_results"]) == 0 and state["attempts"] < 2:
        print("\n[Decision] No results found. Retrying search...")
        return "retry"
    else:
        return "proceed"


# ── Step 4: Build the Graph ───────────────────────────────────────────────────
# StateGraph takes your State class as input.
# .add_node() registers each function as a named node.
# .add_edge() connects nodes in sequence.
# .add_conditional_edges() connects nodes with branching logic.

graph_builder = StateGraph(HotelSearchState)

# Register nodes
graph_builder.add_node("search", search_node)
graph_builder.add_node("rank", ranking_node)
graph_builder.add_node("report", report_node)

# Set entry point — where the graph starts
graph_builder.set_entry_point("search")

# Normal edge: after search, always go to rank
graph_builder.add_edge("search", "rank")

# Conditional edge: after rank, decide where to go
graph_builder.add_conditional_edges(
    "rank",                          # from this node
    should_retry,                    # call this function to decide
    {
        "retry": "search",           # if function returns "retry" → go to search again
        "proceed": "report"          # if function returns "proceed" → go to report
    }
)

# Normal edge: after report, end the graph
graph_builder.add_edge("report", END)

# Compile the graph — this validates everything and prepares it to run
graph = graph_builder.compile()


# ── Step 5: Run the graph ─────────────────────────────────────────────────────
# .invoke() runs the full graph from the entry point.
# We pass the initial state values — only the input fields need values.
# LangGraph fills the rest as nodes run.

print("=" * 50)
print("TEST 1: Normal search with results")
print("=" * 50)

initial_state = {
    "city": "Alexandria",
    "budget": 50.0,
    "search_results": [],
    "ranked_results": [],
    "final_report": "",
    "attempts": 0
}

result = graph.invoke(initial_state)

print("\n" + "=" * 50)
print("FINAL REPORT:")
print("=" * 50)
print(result["final_report"])


# ── Bonus: Print the graph structure as ASCII ─────────────────────────────────
print("\n--- Graph Structure ---")
graph.get_graph().print_ascii()