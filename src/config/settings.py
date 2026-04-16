import os
from dataclasses import dataclass
from dotenv import load_dotenv

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


def get_settings() -> Settings:
    """Returns a Settings instance. Import and call this everywhere."""
    return Settings()