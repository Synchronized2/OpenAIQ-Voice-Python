"""Measure the configured LLM/TTS pipeline without recording the microphone."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame

import config
from chat import ChatSession
from diagnostics import configure_logging
from tts import EdgeSpeaker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=("auto", "none", "low", "medium", "high"))
    parser.add_argument("--prompt", default="hello hello")
    parser.add_argument("--play", action="store_true", help="Play the generated audio; otherwise only validate decoding.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        preferences = json.loads((root / "ui-settings.json").read_text(encoding="utf-8"))
    except (ValueError, OSError):
        preferences = {}
    selected = preferences.get("model") if isinstance(preferences, dict) else None
    model = args.model or (selected if isinstance(selected, str) and selected else config.CHAT_MODEL)
    if args.effort:
        config.CHAT_REASONING_EFFORT = args.effort
    configure_logging()
    report = {"model": model, "reasoning_effort": config.CHAT_REASONING_EFFORT, "audio_played": args.play}
    stopped = threading.Event()
    audio_segments = []
    started = time.perf_counter()

    class ProbeSpeaker(EdgeSpeaker):
        def _play(self, path: Path) -> None:
            duration = pygame.mixer.Sound(file=str(path)).get_length()
            audio_segments.append({
                "ready_seconds": round(time.perf_counter() - started, 3),
                "duration_seconds": round(duration, 3),
                "bytes": path.stat().st_size,
            })
            if args.play:
                super()._play(path)

    speaker = None
    timer = threading.Timer(60.0, stopped.set)
    timer.daemon = True
    chat = None
    exit_code = 0
    try:
        chat = ChatSession(model=model)
        speaker = ProbeSpeaker()
        started = time.perf_counter()
        timer.start()
        speaker.begin_response()
        _, interrupted = chat.reply(args.prompt, on_sentence=speaker.enqueue, should_stop=stopped.is_set)
        if interrupted or not speaker.wait_interruptible(stopped):
            report["error_type"] = "CancelledOrDeadlineExceeded"
            exit_code = 1
    except (Exception, KeyboardInterrupt) as exc:
        report["error_type"] = type(exc).__name__
        exit_code = 1
    finally:
        timer.cancel()
        stopped.set()
        if speaker is not None:
            speaker.close()
    report["llm"] = getattr(chat, "last_timings", {})
    report["audio_segments"] = audio_segments
    report["first_audio_seconds"] = audio_segments[0]["ready_seconds"] if audio_segments else None
    report["total_seconds"] = round(time.perf_counter() - started, 3)
    folder = root / "logs"
    folder.mkdir(exist_ok=True)
    path = folder / f"latency-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    payload = json.dumps(report, ensure_ascii=True, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"Report: {path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
