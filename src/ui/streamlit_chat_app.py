"""
Streamlit Chat UI — conversational interface for the Hotel Booking AI Agent.
Run with: python -m streamlit run src/ui/streamlit_chat_app.py
"""

import streamlit as st
from src.agents.parser_agent import ParserAgent
from src.graph.booking_graph import BookingGraph
from src.config.settings import get_settings


# ── Greeting shown when the chat first opens ──────────────────────────────────
WELCOME_MESSAGE = (
    "👋 Hello! I'm your hotel booking assistant.\n\n"
    "Just tell me what you're looking for — for example:\n"
    "- *\"I want hotels in Cairo under $150 per night\"*\n"
    "- *\"Find me the best 3 hotels in Luxor\"*\n"
    "- *\"Hotels in Hurghada from August 1st to August 5th\"*"
)


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=title,
        page_icon="🏨",
        layout="centered",
    )


def initialize_session() -> None:
    """Sets up session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
    if "search_params" not in st.session_state:
        st.session_state.search_params = {}
    if "searching" not in st.session_state:
        st.session_state.searching = False


def render_message_history() -> None:
    """Renders all messages in the chat history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def add_message(role: str, content: str) -> None:
    """Adds a message to history and renders it immediately."""
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


def render_hotel_results(result: dict) -> None:
    """Renders hotel results as a nicely formatted chat message."""
    lines = [result["final_report"]]

    if result.get("ranked_hotels"):
        lines.append("\n---\n**📋 Full results:**\n")
        for i, hotel in enumerate(result["ranked_hotels"], 1):
            name = hotel.get("name", "Unknown")
            price = f"${hotel['price_per_night']}/night" if hotel.get("price_per_night") else "Price N/A"
            rating = f"⭐ {hotel['rating']}/5" if hotel.get("rating") else ""
            notes = hotel.get("notes", "")
            lines.append(f"**{i}. {name}** — {price} {rating}")
            if notes:
                lines.append(f"   _{notes}_")

    lines.append("\n---\n_Want to search again? Just tell me another city or budget!_")

    return "\n".join(lines)


def reset_search() -> None:
    """Clears search params after a completed search."""
    st.session_state.search_params = {}
    # Keep only the last assistant result message + a fresh prompt
    # so the user can start a new search cleanly
    st.session_state.messages = [
        {"role": "assistant", "content": "✅ Search complete! Feel free to search again — just tell me a new city, budget, or dates."}
    ]


def run_search(params: dict) -> None:
    """Runs the full BookingGraph and adds results to chat."""
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching hotels and reasoning with AI..."):
            try:
                graph = BookingGraph()
                result = graph.run(
                    city=params["city"],
                    budget=float(params["budget"]),
                    check_in=params["check_in"],
                    check_out=params["check_out"],
                )
                response = render_hotel_results(result)

            except Exception as e:
                response = f"Sorry, something went wrong: {str(e)}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    reset_search()


def is_greeting_or_off_topic(text: str) -> bool:
    """Returns True if the message is clearly not a hotel search request."""
    greetings = ["hello", "hi", "hey", "good morning", "good evening", "howdy", "what's up"]
    cleaned = text.lower().strip().rstrip("!.,?")
    return cleaned in greetings


def handle_user_message(user_input: str) -> None:
    """
    Core logic: parse the conversation, decide what to do next.
    Either ask for missing info or run the search.
    """
    add_message("user", user_input)

    # Handle greetings without running the parser
    if is_greeting_or_off_topic(user_input):
        add_message(
            "assistant",
            "👋 Hello! Tell me which city you'd like to search hotels in, "
            "your budget per night, and your travel dates — and I'll find the best options for you!"
        )
        return

    # Run the parser on the full conversation history
    parser = ParserAgent()
    parsed = parser.run(st.session_state.messages)

    if parsed["ready_to_search"]:
        st.session_state.search_params = parsed
        run_search(parsed)
    else:
        question = parsed.get("clarification_question") or (
            "Could you also tell me: " + ", ".join(parsed["missing"]) + "?"
        )
        add_message("assistant", question)


def main() -> None:
    settings = get_settings()
    configure_page(settings.app_title)

    st.title("🏨 Hotel Booking Assistant")
    st.caption("Powered by LangGraph · DeepSeek-R1 / Llama 3.3 · Local & Cloud")

    initialize_session()
    render_message_history()

    # Chat input box (always visible at bottom)
    user_input = st.chat_input("Tell me what hotels you're looking for...")

    if user_input:
        handle_user_message(user_input)


if __name__ == "__main__":
    main()