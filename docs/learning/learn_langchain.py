"""
LangChain Learning File — Phase 2A
This file is for learning only. It will not be part of the final project.
Run it with: python learn_langchain.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Step 1: Connect to your local LLM ────────────────────────────────────────
# ChatOllama speaks to Ollama running on your machine.
# base_url points to where Ollama listens (default port 11434).
# model is the exact name from `ollama list`.

llm = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://localhost:11434",
    temperature=0.3,   # lower = more focused, less random
)


# ── Step 2: Create a prompt template ─────────────────────────────────────────
# {city} and {budget} are placeholders filled at runtime.
# SystemMessage sets the agent's role/persona.
# HumanMessage is what the user says.

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful hotel booking assistant. Be concise."),
    ("human", "Find me the best hotels in {city} with a budget of {budget} USD per night."),
])


# ── Step 3: Output parser ─────────────────────────────────────────────────────
# StrOutputParser simply takes the LLM's response and returns it as a clean string.
# Later we will use parsers that return structured JSON or Python objects.

parser = StrOutputParser()


# ── Step 4: Chain them together using the pipe operator ───────────────────────
# This is LCEL (LangChain Expression Language).
# prompt → llm → parser is read left to right, just like a pipeline.

chain = prompt | llm | parser


# ── Step 5: Invoke the chain ──────────────────────────────────────────────────
# .invoke() runs the full pipeline with the given inputs.
# The dict keys must match the {placeholders} in the prompt template.

print("Sending request to DeepSeek-R1... (first run may be slow)\n")

response = chain.invoke({
    "city": "Alexandria, Egypt",
    "budget": "50"
})

print("Response:")
print(response)



# ── Bonus: Structured output with JSON ────────────────────────────────────────
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a hotel data extractor. 
Always respond with ONLY valid JSON, no extra text.
Format: {{"hotels": [{{"name": "...", "price_per_night": 0, "rating": 0.0}}]}}"""),
    ("human", "List 2 example hotels in {city} under {budget} USD per night."),
])

json_parser = JsonOutputParser()

json_chain = json_prompt | llm | json_parser

print("\n--- Structured JSON output ---")
json_response = json_chain.invoke({
    "city": "Cairo",
    "budget": "80"
})

print(type(json_response))   # should be <class 'dict'>
print(json_response)