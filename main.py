#!/usr/bin/env python3
"""AI News Podcast -- CLI entry point.

Pipeline: RSS ingestion -> two-pass LLM synthesis -> edge-tts voice-over ->
audio compression -> Telegram delivery.

Usage:
    python main.py --setup              Interactive first-run setup wizard
    python main.py --run                Run the full pipeline once
    python main.py --run --cadence weekly   Override the configured cadence for this run
    python main.py --run --dry-run      Run the pipeline but skip the Telegram send
                                         (script + audio are still written to ./output)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from podcast.audio_generator import AudioGenerationError, AudioGenerator  # noqa: E402
from podcast.config import Config, ConfigError  # noqa: E402
from podcast.feed_fetcher import FeedFetcher  # noqa: E402
from podcast.script_writer import ScriptWriter  # noqa: E402
from podcast.setup_wizard import run_setup  # noqa: E402
from podcast.telegram_notifier import TelegramDeliveryError, TelegramNotifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_news_podcast")


def run_pipeline(config: Config, dry_run: bool = False) -> int:
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1/4: Fetching news (cadence=%s) ===", config.cadence)
    fetcher = FeedFetcher()
    articles = fetcher.fetch_recent(cadence=config.cadence)
    if not articles:
        logger.error("No articles were found across any feed. Nothing to synthesize; exiting.")
        return 1
    if len(articles) > config.max_articles:
        logger.info(
            "Capping %d fetched articles down to the %d most recent (set MAX_ARTICLES in .env to change).",
            len(articles),
            config.max_articles,
        )
        articles = articles[: config.max_articles]
    logger.info("Collected %d article(s) to synthesize.", len(articles))

    articles_path = output_dir / "fetched_articles.json"
    articles_path.write_text(
        json.dumps([{**asdict(a), "published": a.published.isoformat()} for a in articles], indent=2),
        encoding="utf-8",
    )
    logger.info("Fetched articles written -> %s", articles_path)

    logger.info("=== Step 2/4: Writing podcast script (provider=%s, 1 request) ===", config.llm_provider)
    try:
        writer = ScriptWriter(
            provider=config.llm_provider,
            gemini_api_key=config.gemini_api_key,
            gemini_model=config.gemini_model,
            groq_api_key=config.groq_api_key,
            groq_model=config.groq_model,
            requests_per_minute=config.gemini_requests_per_minute,
        )
        script_text = writer.write_script(articles)
    except Exception:
        logger.exception("Script synthesis failed.")
        return 1
    script_path = output_dir / "script.txt"
    script_path.write_text(script_text, encoding="utf-8")
    logger.info("Script written (%d words) -> %s", len(script_text.split()), script_path)

    logger.info("=== Step 3/4: Generating voice-over ===")
    try:
        audio_gen = AudioGenerator(
            voice=config.edge_tts_voice,
            target_bitrate_kbps=config.target_bitrate_kbps,
            max_upload_mb=config.max_upload_mb,
        )
        audio_path = audio_gen.generate(script_text, output_dir)
    except AudioGenerationError:
        logger.exception("Audio generation failed.")
        return 1
    logger.info("Audio ready -> %s", audio_path)

    if dry_run:
        logger.info("Dry run: skipping Telegram delivery. Script and audio are in ./output.")
        return 0

    logger.info("=== Step 4/4: Delivering to Telegram ===")
    try:
        notifier = TelegramNotifier(bot_token=config.telegram_bot_token, chat_id=config.telegram_chat_id)
        notifier.send_voice(audio_path, headline_articles=articles)
    except TelegramDeliveryError:
        logger.exception("Telegram delivery failed.")
        return 1

    logger.info("Done. Episode delivered to Telegram chat %s.", config.telegram_chat_id)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and deliver an AI news podcast to Telegram.")
    parser.add_argument("--setup", action="store_true", help="Run the interactive setup wizard.")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline once.")
    parser.add_argument(
        "--cadence",
        choices=["daily", "weekly"],
        default=None,
        help="Override the configured cadence for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but skip sending to Telegram (writes ./output/script.txt and audio).",
    )
    args = parser.parse_args()

    if not args.setup and not args.run:
        parser.print_help()
        return 0

    if args.setup:
        run_setup(PROJECT_ROOT)
        if not args.run:
            return 0

    try:
        config = Config.load(cadence_override=args.cadence)
    except ConfigError as exc:
        logger.error(str(exc))
        return 1

    return run_pipeline(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
