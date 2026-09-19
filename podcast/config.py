"""Configuration loading for the AI News Podcast pipeline.

Reads settings from a local .env file (via python-dotenv) and/or the
process environment. Environment variables always take precedence, which
is what makes this work unmodified inside GitHub Actions (where secrets
are injected as env vars, not as a .env file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

VALID_CADENCES = ("daily", "weekly")
VALID_LLM_PROVIDERS = ("gemini", "groq")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Config:
    telegram_bot_token: str
    telegram_chat_id: str
    llm_provider: str = "gemini"
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    cadence: str = "daily"
    gemini_model: str = "gemini-3.6-flash"
    groq_model: str = "openai/gpt-oss-120b"
    edge_tts_voice: str = "en-US-ChristopherNeural"
    target_bitrate_kbps: int = 48
    max_upload_mb: int = 49  # stay safely under Telegram's 50MB bot limit
    max_articles: int = 6  # single-call prompt stays small and within free-tier quotas
    gemini_requests_per_minute: int = 5  # free-tier default for gemini-3.6-flash
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @classmethod
    def load(cls, env_path: Path | None = None, cadence_override: str | None = None) -> "Config":
        env_path = env_path or ENV_PATH
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            # Still allow plain environment variables (e.g. GitHub Actions secrets)
            load_dotenv(override=False)

        def _require(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise ConfigError(
                    f"Missing required setting '{name}'. Run `python main.py --setup` "
                    "to configure the pipeline, or set it as an environment variable."
                )
            return value

        cadence = (cadence_override or os.environ.get("PODCAST_CADENCE", "daily")).strip().lower()
        if cadence not in VALID_CADENCES:
            raise ConfigError(f"cadence must be one of {VALID_CADENCES}, got '{cadence}'")

        llm_provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
        if llm_provider not in VALID_LLM_PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of {VALID_LLM_PROVIDERS}, got '{llm_provider}'")

        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip() or None
        if llm_provider == "gemini" and not gemini_api_key:
            raise ConfigError(
                "Missing required setting 'GEMINI_API_KEY' (LLM_PROVIDER=gemini). Run "
                "`python main.py --setup` to configure the pipeline, or set it as an environment variable."
            )
        if llm_provider == "groq" and not groq_api_key:
            raise ConfigError(
                "Missing required setting 'GROQ_API_KEY' (LLM_PROVIDER=groq). Run "
                "`python main.py --setup` to configure the pipeline, or set it as an environment variable."
            )

        return cls(
            telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_require("TELEGRAM_CHAT_ID"),
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            groq_api_key=groq_api_key,
            cadence=cadence,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash",
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b",
            edge_tts_voice=os.environ.get("EDGE_TTS_VOICE", "en-US-ChristopherNeural").strip()
            or "en-US-ChristopherNeural",
            target_bitrate_kbps=int(os.environ.get("TARGET_BITRATE_KBPS", "48") or 48),
            max_articles=int(os.environ.get("MAX_ARTICLES", "6") or 6),
            gemini_requests_per_minute=int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "5") or 5),
        )
