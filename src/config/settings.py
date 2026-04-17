import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_ollama import ChatOllama 

load_dotenv()


@dataclass
class Settings:
    """
    Central configuration class.
    All environment variables are loaded here.
    No other file should call os.getenv() directly.
    """
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
    app_title: str = os.getenv("APP_TITLE", "Hotel Booking AI Agent")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))

    def build_llm(self, temperature: float = 0.3) -> ChatOllama:
        """
        Builds and returns a ChatOllama instance using current settings.
        All agents call this instead of creating their own ChatOllama.
        temperature: 0.0 = deterministic, 1.0 = creative. Agents use low values.
        """
        return ChatOllama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=temperature,
        )


def get_settings() -> Settings:
    """Returns a Settings instance. Import and call this everywhere."""
    return Settings()