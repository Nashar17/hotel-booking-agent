---
title: Hotel Booking AI Agent
emoji: 🏨
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.45.0"
python_version: "3.11"
app_file: app.py
pinned: false
---


# 🏨 Hotel Booking AI Agent

🚀 **Live Demo:** [huggingface.co/spaces/Nashar17/hotel-booking-agent](https://huggingface.co/spaces/Nashar17/hotel-booking-agent)

A 100% local multi-agent hotel booking assistant powered by DeepSeek-R1, LangGraph, LangChain, and Streamlit.

The system uses three specialized AI agents that work together to search the web, rank hotels by value, and generate a friendly recommendation — all running on your own machine with no paid APIs.

---

## How it works

## How it works

\```
User types naturally → Parser Agent → extracts city, budget, dates
                            ↓
                    (asks for missing info if needed)
                            ↓
                   Search Agent → Ranking Agent → Report Agent → Chat response
                       ↑                |
                       └── retry if no results (LangGraph)
\```

1. **Parser Agent** — reads the conversation and extracts city, budget, and dates from natural language. Asks follow-up questions for anything missing.
2. **Search Agent** — builds a search query and fetches real hotel data from the web
3. **Ranking Agent** — uses LLM to extract and score hotels by value
4. **Report Agent** — generates a friendly, human-readable recommendation

---

## Tech stack

| Tool | Purpose |
|---|---|
| LangChain | Prompt templates, chains, tool wrappers |
| LangGraph | Multi-agent orchestration and retry logic |
| Ollama + DeepSeek-R1 | Local LLM inference (development) |
| Groq + Llama 3.3 70B | Cloud LLM inference (deployment) |
| DuckDuckGo / Tavily | Web search — no paid API needed locally |
| Streamlit | Conversational chat UI |
| Docker | Containerization |

---

## Project structure

```bash
hotel-booking-agent/
├── src/
│   ├── agents/
│   │   ├── search_agent.py       # Builds query and fetches web results
│   │   ├── ranking_agent.py      # Extracts and scores hotels with LLM
│   │   ├── report_agent.py       # Generates final recommendation
│   │   └── parser_agent.py         
│   ├── tools/
│   │   └── browser_tool.py       # DuckDuckGo search as LangChain tool
│   ├── graph/
│   │   └── booking_graph.py      # LangGraph state machine
│   ├── config/
│   │   └── settings.py           # Central config loaded from .env
│   └── ui/
│       ├── streamlit_app.py      # Original form-based UI (kept for reference)
│       └── streamlit_chat_app.py # Conversational chat interface (main UI)
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

## Conversational chat UI 

```bash
python -m streamlit run src/ui/streamlit_chat_app.py
```

###  Streamlit UI 

```bash
python -m streamlit run src/ui/streamlit_app.py
```

Open your browser at `http://localhost:8501`

### CLI mode

```bash
python main.py
```

###  Docker

```bash
docker compose up --build
```

Open your browser at `http://localhost:8501`

> **Note:** Ollama must be running on your host machine before starting the Docker container.

---

## Known limitations

- Search results depend on DuckDuckGo snippet quality in local mode.
  Cloud deployment uses Tavily for richer results.
- Hotel prices are not always available in search snippets — the
  ranking agent will note this honestly rather than hallucinate.
- Ollama must be running separately when using Docker locally.
- For best results, use major Egyptian cities like Cairo or Luxor.

---

## 👨‍💻 Author
Mohamed El-Nashar