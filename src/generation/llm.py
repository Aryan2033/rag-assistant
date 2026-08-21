"""
src/generation/llm.py
Swappable LLM backend. Uses Gemini if GEMINI_API_KEY is set (cloud/deploy),
otherwise falls back to local Ollama (development). The rest of the codebase
calls generate() and doesn't care which backend answers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import LLM_MODEL

_USE_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))

if _USE_GEMINI:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
else:
    import ollama


def generate(prompt: str, system: str | None = None) -> str:
        if _USE_GEMINI:
            try:
                model = genai.GenerativeModel(
                    GEMINI_MODEL,
                    system_instruction=system,
                    generation_config={"temperature": 0},
                )
                resp = model.generate_content(prompt)
                # Gemini can return no text if it blocks/declines — handle gracefully
                if not getattr(resp, "text", None):
                    return "I don't have that information in the available documents."
                return resp.text.strip()
            except Exception as e:
                # Rate limits, transient API errors, etc. — don't crash the app
                return f"(The assistant is temporarily unavailable — please try again in a moment.)"
        else:
            # Local Ollama
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                options={"temperature": 0},
            )
            return resp["message"]["content"]