"""OpenRouter chat client: retries + optional streaming with one-shot fallback."""
import logging
import time

from openai import OpenAI

from src.config import Settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.api_key)

    def chat(self, messages: list[dict], timeout: int = 60) -> str:
        last_error = None
        for attempt in range(self.settings.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    timeout=timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM attempt %d failed (%s)", attempt + 1, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"LLM call failed after {self.settings.max_retries} attempts: {last_error}"
        ) from last_error

    def chat_stream(self, messages: list[dict], timeout: int = 60):
        """Yields text deltas. Falls back to one-shot on any streaming error."""
        try:
            stream = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=self.settings.temperature,
                timeout=timeout,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            logger.warning("Streaming failed, falling back to one-shot: %s", exc)
            yield self.chat(messages)