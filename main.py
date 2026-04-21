"""
main.py — CLI entry point for the Hotel Booking AI Agent.
For the full UI, run: python -m streamlit run src/ui/streamlit_chat_app.py
For CLI mode: python main.py
"""

from src.graph.booking_graph import BookingGraph
from src.config.settings import setup_logging


def run_booking(city: str, budget: float, check_in: str, check_out: str) -> None:
    """Runs the full booking workflow and prints the result."""
    print(f"\nSearching hotels in {city}...")
    print(f"Budget: ${budget}/night | {check_in} → {check_out}\n")

    graph = BookingGraph()
    result = graph.run(city, budget, check_in, check_out)

    print("\n" + "=" * 55)
    print("FINAL REPORT")
    print("=" * 55)
    print(result["final_report"])

    if result["top_hotel"]:
        print("\nTop pick:")
        print(f"  Name  : {result['top_hotel']['name']}")
        print(f"  Price : ${result['top_hotel'].get('price_per_night', 'N/A')}/night")
        print(f"  Rating: {result['top_hotel'].get('rating', 'N/A')}")
        print(f"  Score : {result['top_hotel'].get('score', 'N/A')}")

    print(f"\nCompleted in {result['attempts']} search attempt(s).")


if __name__ == "__main__":
    from datetime import date, timedelta
    setup_logging()
    check_in = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    check_out = (date.today() + timedelta(days=10)).strftime("%Y-%m-%d")
    run_booking(
        city="Paris",
        budget=150.0,
        check_in=check_in,
        check_out=check_out,
    )