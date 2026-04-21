"""
Streamlit Chat UI — conversational interface for the Hotel Booking AI Agent.
Run with: python -m streamlit run src/ui/streamlit_chat_app.py
"""

import streamlit as st
from src.agents.parser_agent import ParserAgent
from src.graph.booking_graph import BookingGraph
from src.config.settings import get_settings, setup_logging

setup_logging()

# ── Constants ─────────────────────────────────────────────────────────────────

WELCOME_MESSAGE = (
    "👋 Hi! I'm your hotel booking assistant — I can find hotels anywhere in the world.\n\n"
    "Just tell me what you're looking for in plain English. For example:\n"
    "- *\"I need hotels in Paris under $200 per night\"*\n"
    "- *\"Find me the best 3 hotels in Tokyo from July 10th to July 15th\"*\n"
    "- *\"Hotels in New York, budget $120, this weekend\"*\n\n"
    "I'll handle the rest! 🌍"
)

GREETING_WORDS = {
    "hello", "hi", "hey", "howdy", "hiya", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "sup", "greetings", "salaam", "مرحبا", "أهلا",
}

SEARCH_STEPS = [
    ("🔍", "Searching the web for hotels..."),
    ("🧠", "Analysing and ranking results..."),
    ("📝", "Writing your recommendation..."),
]


# ── Page Config ───────────────────────────────────────────────────────────────

def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=title,
        page_icon="🏨",
        layout="centered",
    )


# ── Session State ─────────────────────────────────────────────────────────────

def initialize_session() -> None:
    """Sets up session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
    if "search_params" not in st.session_state:
        st.session_state.search_params = {}
    if "last_parsed" not in st.session_state:
        st.session_state.last_parsed = {}


# ── Message Helpers ───────────────────────────────────────────────────────────

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


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """
    Shows the currently extracted search parameters so the user
    can see what the agent has understood so far.
    Also provides a 'New Search' reset button.
    """
    with st.sidebar:
        st.title("🏨 Hotel Finder")
        st.caption("Powered by LangGraph · Llama 3.3 / DeepSeek-R1")
        st.divider()

        params = st.session_state.last_parsed
        if params:
            st.subheader("📋 What I understood")

            city = params.get("city")
            budget = params.get("budget")
            check_in = params.get("check_in")
            check_out = params.get("check_out")
            missing = params.get("missing", [])

            if city:
                st.success(f"📍 **City:** {city}")
            else:
                st.warning("📍 City: _not yet mentioned_")

            if budget:
                st.success(f"💰 **Budget:** up to ${budget}/night")
            else:
                st.warning("💰 Budget: _not yet mentioned_")

            if check_in:
                st.success(f"📅 **Check-in:** {check_in}")
            else:
                st.warning("📅 Check-in: _not yet mentioned_")

            if check_out:
                st.success(f"📅 **Check-out:** {check_out}")
            else:
                st.warning("📅 Check-out: _not yet mentioned_")

            if missing:
                st.info(f"Still needed: **{', '.join(missing)}**")

        st.divider()

        if st.button("🔄 New Search", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]
            st.session_state.search_params = {}
            st.session_state.last_parsed = {}
            st.rerun()

        st.divider()
        st.caption("💡 **Tips:**")
        st.caption("- Works for any city worldwide")
        st.caption("- Natural language dates work (e.g. 'next Friday')")
        st.caption("- You can always ask to search again")


# ── Result Rendering ──────────────────────────────────────────────────────────

def render_hotel_results(result: dict) -> str:
    """
    Formats hotel results into a rich Markdown chat message.
    Preserves the full ranked list below the AI summary.
    """
    lines = [result["final_report"]]

    if result.get("ranked_hotels"):
        lines.append("\n---\n")

        # Top pick highlight
        top = result["ranked_hotels"][0]
        top_name = top.get("name", "Unknown")
        top_price = f"${top['price_per_night']}/night" if top.get("price_per_night") else "Price N/A"
        top_rating = f"⭐ {top['rating']}/5" if top.get("rating") else ""
        top_notes = top.get("notes", "")

        lines.append(f"### 🏆 Top Pick: {top_name}")
        lines.append(f"**{top_price}** {top_rating}")
        if top_notes:
            lines.append(f"_{top_notes}_")
        lines.append("")

        # Full list (2nd and beyond)
        runners_up = result["ranked_hotels"][1:]
        if runners_up:
            lines.append("**📋 Other options:**\n")
            for i, hotel in enumerate(runners_up, 2):
                name = hotel.get("name", "Unknown")
                price = f"${hotel['price_per_night']}/night" if hotel.get("price_per_night") else "Price N/A"
                rating = f"⭐ {hotel['rating']}/5" if hotel.get("rating") else ""
                notes = hotel.get("notes", "")
                lines.append(f"\n**{i}. {name}**")
                lines.append(f"{price} {rating}")
                if notes:
                    lines.append(f"_{notes}_")

        attempts = result.get("attempts", 1)
        if attempts > 1:
            lines.append(f"\n_Found after {attempts} search attempts._")

    lines.append("\n---\n_Want to search again? Use the **🔄 New Search** button or just tell me another city!_")
    return "\n".join(lines)


# ── Search Execution ──────────────────────────────────────────────────────────

def run_search(params: dict) -> None:
    """
    Runs the full BookingGraph with live step-by-step progress feedback,
    then appends the result to chat history (does NOT wipe history).
    """
    with st.chat_message("assistant"):
        # Show animated step indicators
        progress_placeholder = st.empty()

        for icon, label in SEARCH_STEPS:
            progress_placeholder.markdown(f"{icon} _{label}_")

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
            response = (
                f"Sorry, something went wrong during the search: `{str(e)}`\n\n"
                "Please try again or use the **🔄 New Search** button."
            )

        progress_placeholder.empty()
        st.markdown(response)

    # Append result to history — no wipe
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.search_params = {}


# ── Greeting Detection ────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    """
    Returns True if the message is clearly just a greeting with no search intent.
    Handles multi-word greetings like "good morning!" and "what's up".
    """
    cleaned = text.lower().strip().rstrip("!.,?")
    # Check exact match and partial match for common multi-word greetings
    if cleaned in GREETING_WORDS:
        return True
    for phrase in GREETING_WORDS:
        if cleaned.startswith(phrase) and len(cleaned) <= len(phrase) + 5:
            return True
    return False


# ── Core Message Handler ──────────────────────────────────────────────────────

def handle_user_message(user_input: str) -> None:
    """
    Core logic: parse the conversation, decide what to do next.
    Either greet, ask for missing info, or run the search.
    """
    add_message("user", user_input)

    if is_greeting(user_input):
        # Count how many messages are already in history
        # If only the welcome message exists, this is the first greeting → respond warmly
        # If there are more messages, the user is just saying hi mid-conversation → nudge them
        existing_messages = len(st.session_state.messages)
        if existing_messages <= 2:
            add_message(
                "assistant",
                "👋 Hello! Great to have you here.\n\n"
                "Tell me which city you'd like to stay in, your budget per night, "
                "and your travel dates — and I'll find the best hotel options for you! 🌍",
            )
        else:
            add_message(
                "assistant",
                "😊 Still here! Just tell me a city, budget, and dates and I'll get searching.",
            )
        return

    # Run the parser on full conversation history
    parser = ParserAgent()
    parsed = parser.run(st.session_state.messages)

    # Update sidebar display
    st.session_state.last_parsed = parsed

    if parsed["ready_to_search"]:
        st.session_state.search_params = parsed
        run_search(parsed)
    else:
        question = parsed.get("clarification_question") or (
            "Could you also share: **" + "**, **".join(parsed["missing"]) + "**?"
        )
        add_message("assistant", question)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()
    configure_page(settings.app_title)

    st.title("🏨 Hotel Booking Assistant")
    st.caption("Find hotels anywhere in the world — just tell me what you need.")

    initialize_session()
    render_sidebar()
    render_message_history()

    user_input = st.chat_input("Where would you like to stay?")
    if user_input:
        handle_user_message(user_input)


if __name__ == "__main__":
    main()