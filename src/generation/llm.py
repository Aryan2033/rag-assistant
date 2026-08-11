"""
src/generation/llm.py — Session 3
Thin wrapper around Ollama so the rest of the code doesn't care which LLM we use.
Swap LLM_MODEL (or this function) later to use an API instead — nothing else changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import LLM_MODEL

import ollama


def generate(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0},  # deterministic: same input -> same answer
    )
    return response["message"]["content"]