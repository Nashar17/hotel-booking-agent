"""
app.py — Hugging Face Spaces entry point.
HF Spaces looks for app.py by default for Streamlit apps.
"""

from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()