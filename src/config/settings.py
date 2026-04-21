"""
Central configuration — single source of truth for all settings.
Automatically switches between local Ollama and Groq cloud
based on the USE_GROQ environment variable.
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

__version__ = "2.0.0"
__author__ = "Mohamed El-Nashar"


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """
    Configures the root logger for the whole application.
    Call this once at startup (main.py and app.py).

    - DEBUG level in development (DEBUG=true in .env)
    - INFO level in production
    - Format: timestamp | level | logger name | message
    """
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ── Settings ──────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    All environment variables are loaded here.
    No other file should call os.getenv() directly.

    LLM instance is built once and cached — call build_llm() as many
    times as you like; it only creates the client on the first call.
    """

    # Ollama settings (local development)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
    )

    # Groq settings (cloud deployment)
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # SerpApi settings (hotel search — replaces DuckDuckGo)
    serpapi_key: str = field(
        default_factory=lambda: os.getenv("SERPAPI_KEY", "")
    )

    # Which LLM backend to use — set USE_GROQ=true in cloud environment
    use_groq: bool = field(
        default_factory=lambda: os.getenv("USE_GROQ", "false").lower() == "true"
    )

    # General app settings
    app_title: str = field(
        default_factory=lambda: os.getenv("APP_TITLE", "Hotel Booking AI Agent")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    )
    search_timeout: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_TIMEOUT", "30"))
    )

    # Private cache — LLM instance is built once and reused
    _llm_cache: dict = field(default_factory=dict, repr=False)

    def build_llm(self, temperature: float = 0.3):
        """
        Builds and returns the appropriate LLM based on environment.
        Result is cached by temperature — subsequent calls with the same
        temperature return the existing instance with no overhead.

        Locally → ChatOllama
        Cloud   → ChatGroq
        """
        logger = logging.getLogger(__name__)

        cache_key = f"{temperature}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        if self.use_groq:
            from langchain_groq import ChatGroq
            logger.info("Using Groq cloud LLM: %s (temp=%.1f)", self.groq_model, temperature)
            llm = ChatGroq(
                model=self.groq_model,
                api_key=self.groq_api_key,
                temperature=temperature,
            )
        else:
            from langchain_ollama import ChatOllama
            logger.info("Using local Ollama LLM: %s (temp=%.1f)", self.ollama_model, temperature)
            llm = ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=temperature,
            )

        self._llm_cache[cache_key] = llm
        return llm


def get_settings() -> Settings:
    """
    Returns a Settings instance.
    Import and call this everywhere — never call os.getenv() directly.
    """
    return Settings()