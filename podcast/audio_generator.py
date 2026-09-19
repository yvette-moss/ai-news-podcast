"""Text-to-speech generation and audio post-processing.

Uses edge-tts (free, no API key required) to synthesize the script, then
transcodes the result with ffmpeg (via pydub) to a low-bitrate OGG/Opus
file. OGG/Opus is what Telegram's Bot API requires for a message to render
as an actual voice-message bubble (waveform + inline playback) via
sendVoice, rather than a generic audio attachment -- which is the
"voice format" behavior the project wants. A low bitrate (default 48kbps,
mono) also keeps even a full 15-minute episode far under Telegram's 50MB
bot upload limit.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import edge_tts
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioGenerationError(RuntimeError):
    pass


class AudioGenerator:
    def __init__(
        self,
        voice: str = "en-US-ChristopherNeural",
        target_bitrate_kbps: int = 48,
        max_upload_mb: int = 49,
    ):
        self.voice = voice
        self.target_bitrate_kbps = target_bitrate_kbps
        self.max_upload_mb = max_upload_mb
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        if shutil.which("ffmpeg") is None:
            raise AudioGenerationError(
                "ffmpeg was not found on PATH. Install it (e.g. `apt-get install ffmpeg` or "
                "`brew install ffmpeg`) -- it's required to compress audio for Telegram."
            )

    def generate(self, script_text: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_mp3 = output_dir / "episode_raw.mp3"
        final_ogg = output_dir / "episode.ogg"

        self._synthesize(script_text, raw_mp3)
        self._compress_to_voice_note(raw_mp3, final_ogg)
        self._validate_size(final_ogg)

        return final_ogg

    def _synthesize(self, script_text: str, out_path: Path) -> None:
        logger.info("Synthesizing speech with edge-tts voice '%s'...", self.voice)

        async def _run():
            communicate = edge_tts.Communicate(script_text, self.voice)
            await communicate.save(str(out_path))

        try:
            asyncio.run(_run())
        except Exception as exc:  # noqa: BLE001
            raise AudioGenerationError(f"edge-tts synthesis failed: {exc}") from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise AudioGenerationError("edge-tts produced an empty audio file")

    def _compress_to_voice_note(self, src_mp3: Path, dst_ogg: Path) -> None:
        logger.info(
            "Transcoding to OGG/Opus at %dkbps mono for Telegram voice delivery...",
            self.target_bitrate_kbps,
        )
        try:
            audio = AudioSegment.from_file(src_mp3)
            audio = audio.set_channels(1)  # voice notes are mono
            audio.export(
                dst_ogg,
                format="ogg",
                codec="libopus",
                bitrate=f"{self.target_bitrate_kbps}k",
            )
        except Exception as exc:  # noqa: BLE001
            raise AudioGenerationError(f"audio compression failed: {exc}") from exc

    def _validate_size(self, path: Path) -> None:
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Final audio size: %.2f MB", size_mb)
        if size_mb > self.max_upload_mb:
            raise AudioGenerationError(
                f"Compressed audio is {size_mb:.1f}MB, over the {self.max_upload_mb}MB safety "
                "limit. Reduce TARGET_BITRATE_KBPS or shorten the script."
            )
