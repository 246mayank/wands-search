"""Two LLM backends behind one call: prompt in, text out. Temperature 0, JSON mode."""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

GEMINI_MODEL = "gemini-3.5-flash-lite"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
RETRIES = 6


class LLMTimeout(Exception):
    pass


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str) -> str: ...


class GeminiClient:
    def __init__(self, model: str = GEMINI_MODEL, sleep: float = 4.0):
        from google import genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.sleep = sleep  # between calls, to stay under the free tier's per-minute limit

    def complete(self, prompt: str) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        for attempt in range(RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                time.sleep(self.sleep)
                return response.text or ""
            except (errors.ServerError, httpx.TimeoutException):
                pass
            except errors.ClientError as error:
                if error.code != 429:
                    raise
            time.sleep(min(90, 5 * 2**attempt))
        raise LLMTimeout(f"{self.model}: gave up after {RETRIES} attempts")


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, timeout: float = 900.0):
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 4096},
        }
        try:
            response = httpx.post(OLLAMA_URL, json=body, timeout=self.timeout)
        except httpx.TimeoutException as error:
            raise LLMTimeout(str(error)) from error
        response.raise_for_status()
        return response.json()["response"]


def make_client(model: str) -> LLMClient:
    return GeminiClient(model) if model.startswith("gemini") else OllamaClient(model)
