"""Delivers the generated episode to Telegram as a voice message."""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from .feed_fetcher import Article

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"
# Telegram's sendVoice caption hard limit (1024) is measured on the rendered/visible
# text, not the raw HTML markup -- so the threshold below must be checked against
# the tag-stripped, entity-decoded string, never the raw <a href="...">...</a> string.
CAPTION_SAFE_LEN = 950  # leave headroom below the hard 1024 limit
MESSAGE_MAX_LEN = 4096  # Telegram's hard limit for sendMessage text
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keycap numeral emojis, used instead of plain "1." "2." for a friendlier-looking
# list. Telegram renders these as small numbered badges. Supports up to 10 items.
_KEYCAP_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


class TelegramDeliveryError(RuntimeError):
    pass


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 120):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds

    def send_voice(self, audio_path: Path, headline_articles: List[Article]) -> None:
        headline_list = self._build_headline_list(headline_articles)
        fits_in_caption = len(self._visible_text(headline_list)) <= CAPTION_SAFE_LEN
        caption = headline_list if fits_in_caption else "Today's AI headlines are linked in the next message ⬇️"

        url = TELEGRAM_API_BASE.format(token=self.bot_token, method="sendVoice")
        with open(audio_path, "rb") as audio_file:
            files = {"voice": (audio_path.name, audio_file, "audio/ogg")}
            data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
            try:
                response = requests.post(url, data=data, files=files, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                raise TelegramDeliveryError(f"Network error sending voice message: {exc}") from exc

        if not response.ok:
            raise TelegramDeliveryError(
                f"Telegram API returned {response.status_code}: {response.text}"
            )

        payload = response.json()
        if not payload.get("ok", False):
            raise TelegramDeliveryError(f"Telegram API rejected the message: {payload}")

        logger.info("Voice message delivered to chat %s", self.chat_id)

        if not fits_in_caption:
            self._send_html_message(headline_list)

    def send_text(self, text: str) -> None:
        """Fallback / auxiliary plain-text message (e.g. for error alerts)."""
        url = TELEGRAM_API_BASE.format(token=self.bot_token, method="sendMessage")
        try:
            response = requests.post(
                url,
                data={"chat_id": self.chat_id, "text": text[:MESSAGE_MAX_LEN]},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to send fallback text message: %s", exc)

    def _send_html_message(self, html_text: str) -> None:
        """Best-effort follow-up (e.g. the headline list when it didn't fit in the caption)."""
        url = TELEGRAM_API_BASE.format(token=self.bot_token, method="sendMessage")
        text = self._truncate_at_line_boundary(html_text, MESSAGE_MAX_LEN)
        try:
            response = requests.post(
                url,
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "link_preview_options": json.dumps({"is_disabled": True}),
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to send follow-up headline list: %s", exc)

    @staticmethod
    def _build_headline_list(articles: List[Article], max_headlines: int = 5) -> str:
        title = f"🎙️ <b>AI News Podcast</b> — {datetime.now():%d.%m.%Y}"
        if not articles:
            return f"{title}\n\nYour AI news podcast is ready. 🎧"

        lines = [title, ""]
        for i, article in enumerate(articles[:max_headlines], start=1):
            keycap = _KEYCAP_EMOJIS[i - 1] if i <= len(_KEYCAP_EMOJIS) else f"{i}."
            headline = html.escape(article.title)
            source = html.escape(article.source)
            href = html.escape(article.link, quote=True)
            lines.append(f'{keycap} <a href="{href}">{headline}</a> 📰 <i>{source}</i>')

        return "\n".join(lines)

    @staticmethod
    def _visible_text(html_text: str) -> str:
        """Approximate what Telegram will actually render: strip tags, decode entities."""
        return html.unescape(_HTML_TAG_RE.sub("", html_text))

    @staticmethod
    def _truncate_at_line_boundary(text: str, max_len: int) -> str:
        """Truncate to the last full line under max_len so an HTML tag is never cut in half."""
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_newline = truncated.rfind("\n")
        return truncated[:last_newline] if last_newline > 0 else truncated
