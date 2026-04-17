"""
main.py — Entry point for the Hotel Booking AI Agent.
Calls the BookingGraph and prints the result.
For the full UI, run: streamlit run src/ui/streamlit_app.py
"""

from src.graph.booking_graph import BookingGraph


def run_booking(city: str, budget: float, check_in: str, check_out: str) -> None:
    """Runs the full booking workflow and prints the result."""
    print(f"\nSearching hotels in {city}, Egypt...")
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
    run_booking(
        city="Cairo",
        budget=300.0,
        check_in="2025-08-01",
        check_out="2025-08-04",
    )