"""Synthesizes fetched articles into a spoken-word podcast script.

Single-call pipeline: every fetched article's title, source, and excerpt is
dumped into one prompt, and the model is asked to pick the most impactful
stories and write the complete conversational script directly. With
MAX_ARTICLES kept small (default 6), the whole bundle comfortably fits in a
single request -- avoiding the per-article map/reduce loop that burns
through free-tier request-per-day quotas in one run.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from .feed_fetcher import Article

logger = logging.getLogger(__name__)

# Patterns providers' error messages use to signal "wait and retry" vs. a
# permanent failure (bad request, bad key, etc.) that retrying won't fix.
_RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503", "DEADLINE_EXCEEDED", "rate_limit")
_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)

# ~15 minutes at a natural spoken pace (~150 wpm) is roughly 2200-2400 words.
TARGET_WORD_COUNT = 2300

SCRIPT_SYSTEM_PROMPT = """You are the writer and host of a daily AI industry news \
podcast. You will be given a bundle of raw article headlines and excerpts pulled \
from today's AI news feeds. First, pick the 5-7 most impactful, newsworthy \
stories from the bundle -- ignore the rest. Then write a single continuous, \
conversational podcast script covering only those stories, meant to be read \
aloud by a text-to-speech voice.

Hard rules:
- Write ONLY the words the host will speak. No markdown, no headers, no bullet \
points, no asterisks, no stage directions, no sound-effect cues, no bracketed \
notes, and no mention of which stories you skipped.
- Open with a brief, warm greeting and a one-line preview of what's covered, \
then a natural sign-off at the end. Do not mention specific dates unless given.
- Use natural spoken transitions between stories ("Speaking of which...", \
"In related news...", "Meanwhile, over at...", "Let's shift gears...").
- Write acronyms and technical terms the way they should be PRONOUNCED, not \
spelled. For example write "L L M" instead of "LLM", "gen-AI" instead of \
"GenAI", "API" as "A P I" only if it's normally spoken letter-by-letter (most \
listeners say "API" as a word so keep it as "API" is fine to leave as-is) -- \
use your judgment for what sounds natural when read by a voice synthesizer, \
favoring clarity over strict phonetic spelling.
- Vary sentence length and rhythm like a real host talking, not a press release.
- Give the listener context on why each story matters, not just what happened.
- Base every claim only on the given excerpts -- never fabricate facts or add \
outside knowledge.
- Target roughly {target_words} words total (about 15 minutes at a natural \
speaking pace). Do not pad with filler -- if there isn't enough material, a \
slightly shorter script is fine.
- Output nothing but the finished script. Do not restate these instructions, \
do not list which stories you chose, and do not add any preamble before the \
host's first word."""


class ScriptWriter:
    def __init__(
        self,
        provider: str = "gemini",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-3.6-flash",
        groq_api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-120b",
        requests_per_minute: int = 5,
        max_retries: int = 5,
    ):
        self.provider = provider
        self.gemini_model = gemini_model
        self.groq_model = groq_model
        self.max_retries = max_retries
        # Only one request is made per run now, but keep a small floor between
        # calls in case a retry follows quickly on a transient error.
        self.min_interval_seconds = (60.0 / max(requests_per_minute, 1)) * 1.1
        self._last_request_at: Optional[float] = None

        if provider == "groq":
            from openai import OpenAI

            self.client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        else:
            from google import genai

            self.client = genai.Client(api_key=gemini_api_key)

    def write_script(self, articles: List[Article]) -> str:
        if not articles:
            raise ValueError("No articles provided to synthesize.")

        bundle = self._build_bundle(articles)
        prompt = (
            f"Here are today's raw article bundle ({len(articles)} articles):\n\n"
            f"{bundle}\n\n"
            "Pick the most impactful stories and write the full podcast script now."
        )
        response_text = self._generate_with_retry(
            prompt=prompt,
            system_instruction=SCRIPT_SYSTEM_PROMPT.format(target_words=TARGET_WORD_COUNT),
            temperature=0.7,
            max_output_tokens=4096,
        )
        text = response_text.strip()
        if not text:
            raise RuntimeError("Model returned an empty podcast script")
        return text

    @staticmethod
    def _build_bundle(articles: List[Article]) -> str:
        entries = []
        for i, article in enumerate(articles):
            entries.append(
                f"{i + 1}. [{article.source}] {article.title}\n"
                f"   Excerpt: {article.summary}"
            )
        return "\n\n".join(entries)

    # -- Shared: rate limiting + retry ------------------------------------

    def _throttle(self) -> None:
        """Sleep as needed so a retry doesn't immediately re-hit a rate limit."""
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.min_interval_seconds - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _generate_with_retry(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempt = 0
        while True:
            self._throttle()
            try:
                if self.provider == "groq":
                    return self._call_groq(prompt, system_instruction, temperature, max_output_tokens)
                return self._call_gemini(prompt, system_instruction, temperature, max_output_tokens)
            except Exception as exc:  # noqa: BLE001 - inspected below, re-raised if not retryable
                attempt += 1
                if not self._is_retryable(exc) or attempt > self.max_retries:
                    raise
                delay = self._extract_retry_delay(exc) or min(2.0 ** attempt, 30.0)
                logger.warning(
                    "%s call hit a transient error (attempt %d/%d); retrying in %.1fs: %s",
                    self.provider,
                    attempt,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

    def _call_gemini(self, prompt: str, system_instruction: str, temperature: float, max_output_tokens: int) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return response.text or ""

    def _call_groq(self, prompt: str, system_instruction: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc)
        return any(marker in message for marker in _RETRYABLE_MARKERS)

    @staticmethod
    def _extract_retry_delay(exc: Exception) -> Optional[float]:
        match = _RETRY_DELAY_RE.search(str(exc))
        if match:
            # Add a small buffer on top of what the API suggests.
            return float(match.group(1)) + 1.0
        return None
