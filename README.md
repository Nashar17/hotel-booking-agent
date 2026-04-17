# 🏨 Hotel Booking AI Agent

A 100% local multi-agent hotel booking assistant powered by DeepSeek-R1, LangGraph, LangChain, and Streamlit.

The system uses three specialized AI agents that work together to search the web, rank hotels by value, and generate a friendly recommendation — all running on your own machine with no paid APIs.

---

## How it works

User Input → Search Agent → Ranking Agent → Report Agent → UI
↑                |
└── retry if no results found (LangGraph)

1. **Search Agent** — builds a query and fetches real hotel data from the web
2. **Ranking Agent** — uses DeepSeek-R1 to extract and score hotels by value
3. **Report Agent** — generates a friendly, human-readable recommendation

---

## Tech stack

| Tool | Purpose |
|---|---|
| DeepSeek-R1 (via Ollama) | Local LLM — reasoning and extraction |
| LangChain | Prompt templates, chains, tool wrappers |
| LangGraph | Multi-agent orchestration and retry logic |
| Streamlit | Web UI |
| DuckDuckGo (ddgs) | Free web search — no API key needed |
| Docker | Containerization |

---

## Project structure

```bash
hotel-booking-agent/
├── src/
│   ├── agents/
│   │   ├── search_agent.py       # Builds query and fetches web results
│   │   ├── ranking_agent.py      # Extracts and scores hotels with LLM
│   │   └── report_agent.py       # Generates final recommendation
│   ├── tools/
│   │   └── browser_tool.py       # DuckDuckGo search as LangChain tool
│   ├── graph/
│   │   └── booking_graph.py      # LangGraph state machine
│   ├── config/
│   │   └── settings.py           # Central config loaded from .env
│   └── ui/
│       └── streamlit_app.py      # Streamlit web interface
├── main.py                       # CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- DeepSeek-R1 pulled: `ollama pull deepseek-r1:7b`
- Docker Desktop (for containerized run)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Nashar17/hotel-booking-agent.git
cd hotel-booking-agent
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
```

Edit `.env` if needed (defaults work out of the box with Ollama).

---

## Running the app

### Option A — Streamlit UI (recommended)

```bash
python -m streamlit run src/ui/streamlit_app.py
```

Open your browser at `http://localhost:8501`

### Option B — CLI mode

```bash
python main.py
```

### Option C — Docker

```bash
docker compose up --build
```

Open your browser at `http://localhost:8501`

> **Note:** Ollama must be running on your host machine before starting the Docker container.

---

## Known limitations

- Search results depend on DuckDuckGo snippet quality. Cairo returns better results than smaller cities.
- Prices are not always available in search snippets — the ranking agent will note this honestly.
- Ollama must be running separately (not included in Docker container).

---

## 👨‍💻 Author
Mohamed El-Nashar