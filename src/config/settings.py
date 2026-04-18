"""
Central configuration — single source of truth for all settings.
Automatically switches between local Ollama and Groq cloud
based on the USE_GROQ environment variable.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """
    All environment variables are loaded here.
    No other file should call os.getenv() directly.
    """
    # Ollama settings (local)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")

    # Groq settings (cloud deployment)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Which backend to use — set USE_GROQ=true in cloud environment
    use_groq: bool = os.getenv("USE_GROQ", "false").lower() == "true"

    # General settings
    app_title: str = os.getenv("APP_TITLE", "Hotel Booking AI Agent")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))

    def build_llm(self, temperature: float = 0.3):
        """
        Builds and returns the appropriate LLM based on environment.
        Locally → ChatOllama
        Cloud   → ChatGroq
        All agents call this — zero changes needed in agent files.
        """
    
        if self.use_groq:
            from langchain_groq import ChatGroq
            print(f"[Settings] Using Groq cloud LLM: {self.groq_model}")
            return ChatGroq(
                model=self.groq_model,
                api_key=self.groq_api_key,
                temperature=temperature,
            )
        else:
            from langchain_ollama import ChatOllama
            print(f"[Settings] Using local Ollama LLM: {self.ollama_model}")
            return ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=temperature,
            )


def get_settings() -> Settings:
    """Returns a Settings instance. Import and call this everywhere."""
    return Settings()