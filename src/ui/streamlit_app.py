"""
Streamlit UI — Hotel Booking AI Agent
Run with: streamlit run src/ui/streamlit_app.py
"""

import streamlit as st
from datetime import date, timedelta
from src.graph.booking_graph import BookingGraph
from src.config.settings import get_settings


def configure_page(title: str) -> None:
    """Sets Streamlit page config. Must be the first Streamlit call."""
    st.set_page_config(
        page_title=title,
        page_icon="🏨",
        layout="centered",
    )


def render_header(title: str) -> None:
    """Renders the app title and subtitle."""
    st.title("🏨 " + title)
    st.markdown(
        "Powered by **DeepSeek-R1** running locally via Ollama · "
        "Multi-agent search with **LangGraph**"
    )
    st.divider()


def render_search_form() -> dict | None:
    """
    Renders the search input form.
    Returns a dict of inputs when submitted, None otherwise.
    """
    with st.form("search_form"):
        col1, col2 = st.columns(2)

        with col1:
            city = st.text_input(
                "City",
                value="Cairo",
                placeholder="e.g. Cairo, Luxor, Hurghada",
            )

        with col2:
            budget = st.number_input(
                "Max budget (USD/night)",
                min_value=10,
                max_value=1000,
                value=150,
                step=10,
            )

        col3, col4 = st.columns(2)

        with col3:
            check_in = st.date_input(
                "Check-in date",
                value=date.today() + timedelta(days=7),
                min_value=date.today(),
            )

        with col4:
            check_out = st.date_input(
                "Check-out date",
                value=date.today() + timedelta(days=10),
                min_value=date.today() + timedelta(days=1),
            )

        submitted = st.form_submit_button(
            "🔍 Find Hotels",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if check_out <= check_in:
            st.error("Check-out date must be after check-in date.")
            return None
        return {
            "city": city.strip(),
            "budget": float(budget),
            "check_in": str(check_in),
            "check_out": str(check_out),
        }

    return None


def render_top_hotel(hotel: dict) -> None:
    """Renders a highlighted card for the top recommended hotel."""
    st.subheader("⭐ Top Pick")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hotel", hotel.get("name", "N/A"))
        with col2:
            price = hotel.get("price_per_night")
            st.metric("Price/night", f"${price}" if price else "N/A")
        with col3:
            rating = hotel.get("rating")
            st.metric("Rating", f"{rating}/5" if rating else "N/A")

        notes = hotel.get("notes")
        if notes:
            st.caption(notes)


def render_hotels_table(ranked_hotels: list) -> None:
    """Renders the full ranked hotel list as a table."""
    if not ranked_hotels:
        return

    st.subheader("📋 All Results")

    table_data = []
    for i, hotel in enumerate(ranked_hotels, 1):
        table_data.append({
            "Rank": i,
            "Hotel": hotel.get("name", "Unknown"),
            "Price/night": f"${hotel['price_per_night']}" if hotel.get("price_per_night") else "N/A",
            "Rating": f"{hotel['rating']}/5" if hotel.get("rating") else "N/A",
            "Score": hotel.get("score", "N/A"),
            "Notes": hotel.get("notes", ""),
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def render_results(result: dict) -> None:
    """Renders the full results section after the graph runs."""
    st.divider()

    # Show attempt count as info
    attempts = result.get("attempts", 1)
    if attempts > 1:
        st.info(f"ℹ️ Search completed after {attempts} attempts.")

    # Final report from ReportAgent
    st.subheader("📝 Recommendation")
    st.markdown(result["final_report"])

    # Top hotel card
    if result.get("top_hotel"):
        render_top_hotel(result["top_hotel"])

    # Full table
    if result.get("ranked_hotels"):
        render_hotels_table(result["ranked_hotels"])


def main() -> None:
    """Main entry point — orchestrates the full UI flow."""
    settings = get_settings()

    configure_page(settings.app_title)
    render_header(settings.app_title)

    inputs = render_search_form()

    if inputs:
        with st.spinner("🤖 Agents are searching the web and reasoning with DeepSeek-R1..."):
            try:
                graph = BookingGraph()
                result = graph.run(
                    city=inputs["city"],
                    budget=inputs["budget"],
                    check_in=inputs["check_in"],
                    check_out=inputs["check_out"],
                )
                st.session_state["last_result"] = result

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.session_state["last_result"] = None

    # Show last result if it exists (persists across reruns)
    if st.session_state.get("last_result"):
        render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()