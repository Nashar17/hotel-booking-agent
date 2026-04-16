"""
Browser Tool — wraps DuckDuckGo search as a LangChain-compatible tool.
Agents call this tool to fetch real hotel data from the web.
"""

from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field


class HotelSearchInput(BaseModel):
    """
    Input schema for the HotelSearchTool.
    Pydantic validates that the agent passes the correct argument types.
    """
    query: str = Field(
        description="Search query for finding hotels. Example: 'best hotels in Alexandria Egypt under 50 USD per night'"
    )


class HotelSearchTool(BaseTool):
    """
    A LangChain tool that searches DuckDuckGo for hotel information.
    Agents call this tool by name when they need real hotel data.
    """

    name: str = "hotel_search"
    description: str = (
        "Use this tool to search for hotels in a specific city with a given budget. "
        "Input should be a natural language query like: "
        "'best hotels in Cairo Egypt under 80 USD per night'. "
        "Returns a list of hotel search results from the web."
    )
    args_schema: type[BaseModel] = HotelSearchInput

    # The DuckDuckGo search instance — created once, reused on every call
    _search: DuckDuckGoSearchRun = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ because Pydantic models are strict about attribute setting
        object.__setattr__(self, "_search", DuckDuckGoSearchRun())

    def _run(self, query: str) -> str:
        """
        Executes the search and returns raw results as a string.
        LangChain calls _run() when an agent uses this tool.
        """
        try:
            results = self._search.run(query)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}. Please try a different query."

    async def _arun(self, query: str) -> str:
        """
        Async version of _run. Required by BaseTool.
        We raise NotImplementedError since we are using sync search for now.
        """
        raise NotImplementedError("HotelSearchTool does not support async yet.")