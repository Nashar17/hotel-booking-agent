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

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-orange)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-green)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-chat_UI-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-blue)](https://huggingface.co/spaces/Nashar17/hotel-booking-agent)

A multi-agent hotel booking assistant that finds hotels **anywhere in the world** using real data from Google Hotels (via SerpApi). Powered by LangGraph, LangChain, Llama 3.3 70B (Groq), and Streamlit.

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Nashar17/hotel-booking-agent)**

---

## ✨ Features

- 🌍 **Worldwide search** — any city, any country
- 💬 **Conversational UI** — natural language, no forms needed
- 💰 **Real hotel prices** — via SerpApi Google Hotels API
- 🤖 **Multi-agent pipeline** — Parser → Search → Rank → Report
- 🔄 **Automatic retry** — relaxes budget by 20% and retries if no results found
- 📋 **Live sidebar** — shows extracted parameters as you type
- 🪵 **Structured logging** — INFO/DEBUG levels, no bare print() calls
- 🏗️ **Clean OOP architecture** — single config source of truth, LangGraph state machine
- 🐳 **Docker ready** — one command deployment

---

## How it works

```
User types naturally → Parser Agent → extracts city, budget, dates
                                    ↓
                         (asks for missing info if needed)
                                    ↓
              Search Agent → Ranking Agent → Report Agent → Chat response
                    ↑               |
                    └── retry with relaxed budget if no results (LangGraph)
```

1. **Parser Agent** — reads the full conversation and extracts city, budget, and dates from natural language. Asks a single follow-up question for any missing fields.
2. **Search Agent** — queries SerpApi Google Hotels with real check-in/check-out dates. On retry, expands the budget ceiling by 20% to surface more candidates.
3. **Ranking Agent** — uses the LLM to validate and score hotels by value: 70% rating weight + 30% budget-savings weight. Filters out hotels with missing or empty names.
4. **Report Agent** — generates a warm, concise recommendation under 200 words based strictly on the data returned — never invents details.

---

## Tech stack

| Tool | Purpose |
|------|---------|
| LangChain | Prompt templates, chains, tool wrappers |
| LangGraph | Multi-agent orchestration and retry logic |
| Ollama + DeepSeek-R1 | Local LLM inference (development) |
| Groq + Llama 3.3 70B | Cloud LLM inference (deployment) |
| SerpApi Google Hotels | Real hotel data with prices and ratings worldwide |
| Streamlit | Conversational chat UI with live sidebar |
| Docker | Containerization |

---

## Project structure

```
hotel-booking-agent/
├── src/
│   ├── agents/
│   │   ├── parser_agent.py       # Extracts city/budget/dates from conversation
│   │   ├── search_agent.py       # Builds query and fetches hotel data
│   │   ├── ranking_agent.py      # Validates, filters, and scores hotels
│   │   └── report_agent.py       # Generates final recommendation
│   ├── tools/
│   │   └── browser_tool.py       # SerpApi Google Hotels (DuckDuckGo fallback)
│   ├── graph/
│   │   └── booking_graph.py      # LangGraph state machine
│   ├── config/
│   │   └. settings.py           # Central config, LLM factory, logging setup
│   └── ui/
│       ├── streamlit_app.py      # Original form-based UI (kept for reference)
│       └── streamlit_chat_app.py # Conversational chat interface (main UI)
├── main.py                       # CLI entry point
├── app.py                        # Hugging Face Spaces entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) — only needed for local LLM (optional if using Groq)
- A free [SerpApi](https://serpapi.com) key — 100 searches/month free, no credit card needed
- A free [Groq](https://console.groq.com) key — for cloud deployment

### 1. Clone the repo

```bash
git clone https://github.com/Nashar17/hotel-booking-agent.git
cd hotel-booking-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
```

Edit `.env` with your keys:

```env
USE_GROQ=true
GROQ_API_KEY=your_groq_key_here
SERPAPI_KEY=your_serpapi_key_here
```

For local development with Ollama:

```env
USE_GROQ=false
OLLAMA_MODEL=deepseek-r1:7b
SERPAPI_KEY=your_serpapi_key_here
```

### 5. Get your SerpApi key (free)

1. Go to [serpapi.com](https://serpapi.com) → sign up (no credit card needed)
2. Copy your API key from the dashboard
3. Add it to `.env` as `SERPAPI_KEY=...`

Free tier: **100 searches/month** — enough for demos and interviews.

---

## Running the app

### Chat UI (recommended)

```bash
python -m streamlit run src/ui/streamlit_chat_app.py
```

### CLI mode

```bash
python main.py
```

### Docker

```bash
docker compose up --build
```

Open: `http://localhost:8501`

---

## Deploying to Hugging Face Spaces

```bash
git remote add huggingface https://huggingface.co/spaces/Nashar17/hotel-booking-agent
git push huggingface main
```

Add secrets in your HF Space → Settings → Variables and Secrets:
- `GROQ_API_KEY`
- `SERPAPI_KEY`
- `USE_GROQ` = `true`

---

## Known limitations

- SerpApi free tier is limited to 100 searches/month
- Without a SerpApi key, falls back to DuckDuckGo (no guaranteed prices)
- Check-in dates must be in the future — SerpApi rejects past dates
- Ollama must be running separately when using Docker locally

---

## 👨‍💻 Author

**Mohamed El-Nashar** — Mechatronics student & AI Engineering enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Nashar17-black?logo=github)](https://github.com/Nashar17)