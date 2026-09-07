"""Sequential, key-safe latency benchmark using the app's actual chat/TTS code."""

from __future__ import annotations

import contextlib
import argparse
import io
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import pygame

import config
from chat import ChatSession
from tts import EdgeSpeaker


MODELS = (
    "deepseek-v4-flash", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.3-codex-spark",
    "qwen3.7-plus", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.2", "gpt-5.4",
    "gpt-5.5", "gpt-5.6", "gpt-5.6-sol", "gpt-5.3-codex", "gpt-6-astra",
    "gpt-reserve", "codex-auto-review",
)
PROMPTS = (
    "你好，请用一句不超过三十个汉字的话，推荐一种下班后放松的活动。",
    "今天有点累，请用一句不超过三十个汉字的话安慰我。",
    "周末想在家休息，请用一句不超过三十个汉字的话推荐一件轻松的事。",
)


def trial(model: str, phase: str, prompt: str, audio: bool = False) -> dict:
    result = {"model": model, "phase": phase, "prompt": prompt,
              "effort": config.CHAT_REASONING_EFFORT if model.startswith("gpt-5") else "auto"}
    started = time.perf_counter()
    stopped = threading.Event()
    timer = threading.Timer(55, stopped.set)
    timer.daemon = True
    speaker = None

    class ProbeSpeaker(EdgeSpeaker):
        def _play(self, path: Path) -> None:
            duration = pygame.mixer.Sound(file=str(path)).get_length()
            if result.get("first_audio_s") is None:
                result["first_audio_s"] = round(time.perf_counter() - started, 3)
                result["first_audio_duration_s"] = round(duration, 3)

    def sentence(text: str):
        if result.get("first_sentence_s") is None:
            result["first_sentence_s"] = round(time.perf_counter() - started, 3)
        if speaker:
            speaker.enqueue(text)

    print(f"START {phase} {model}", flush=True)
    try:
        session = ChatSession(model=model)
        if audio:
            speaker = ProbeSpeaker()
            speaker.begin_response()
        started = time.perf_counter()
        timer.start()
        with contextlib.redirect_stdout(io.StringIO()):
            answer, cancelled = session.reply(prompt, on_sentence=sentence, should_stop=stopped.is_set)
        result["first_token_s"] = round(session.last_timings["first_token"], 3) if session.last_timings.get("first_token") is not None else None
        result["llm_total_s"] = round(session.last_timings["total"], 3)
        result["reply"] = answer
        if cancelled or (speaker and not speaker.wait_interruptible(stopped)):
            raise TimeoutError("Benchmark cancelled at its deadline")
        if audio and result.get("first_audio_s") is None:
            raise RuntimeError("No decodable audio")
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["http_status"] = getattr(exc, "status_code", None)
        message = str(exc).replace(config.chat_api_key(), "[REDACTED]")
        result["error"] = message[:500]
    finally:
        timer.cancel()
        stopped.set()
        if speaker:
            speaker.close()
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    print(json.dumps({k: v for k, v in result.items() if k not in {"reply", "prompt", "error"}}, ensure_ascii=True), flush=True)
    return result


def summary(trials: list[dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        samples = [r for r in trials if r["model"] == model and r["phase"] != "tts"]
        good = [r for r in samples if r["status"] == "ok"]
        row = {"model": model, "attempts": len(samples), "successes": len(good)}
        for field in ("first_token_s", "first_sentence_s", "llm_total_s"):
            values = [r[field] for r in good if r.get(field) is not None]
            row[field] = round(statistics.median(values), 3) if values else None
        audio = [r for r in trials if r["model"] == model and r["phase"] == "tts"]
        if audio:
            decoded = [r for r in audio if r["status"] == "ok"]
            row["tts"] = {"attempts": len(audio), "successes": len(decoded)}
            if decoded:
                values = [r["first_audio_s"] for r in decoded]
                row["tts"].update({
                    "first_audio_median_s": round(statistics.median(values), 3),
                    "first_audio_min_s": min(values), "first_audio_max_s": max(values),
                    "synthesis_delay_median_s": round(statistics.median(r["first_audio_s"] - r["first_sentence_s"] for r in decoded), 3),
                })
        rows.append(row)
    return sorted(rows, key=lambda r: r["first_sentence_s"] if r["first_sentence_s"] is not None else float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-audio", type=Path, help="Append two audio rounds to an existing benchmark report.")
    args = parser.parse_args()
    # Process-local overrides; config.py and ui-settings.json remain untouched.
    config.CHAT_FIRST_TOKEN_TIMEOUT_SECONDS = 20.0
    config.CHAT_TOTAL_TIMEOUT_SECONDS = 35.0
    root = Path(__file__).resolve().parent
    folder = root / "logs"
    folder.mkdir(exist_ok=True)
    path = args.resume_audio or folder / f"model-benchmark-{datetime.now():%Y%m%d-%H%M%S}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if args.resume_audio else {
        "started_at": datetime.now().astimezone().isoformat(), "completed": False,
        "method": "Sequential cold-client requests through ChatSession, fresh conversation each trial; no microphone or playback.",
        "limits": {"first_content_s": 20, "llm_total_s": 35},
        "request": {"max_tokens": config.MAX_TOKENS, "temperature": config.TEMPERATURE,
                    "gpt5_reasoning_effort": config.CHAT_REASONING_EFFORT,
                    "tts_segment_min": config.TTS_SEGMENT_MIN_CHARS, "tts_segment_max": config.TTS_SEGMENT_MAX_CHARS},
        "excluded": {"gpt-image-2": "Image model, not ranked for conversation."},
        "trials": [],
    }

    def save():
        report["summary"] = summary(report["trials"])
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"REPORT {path}", flush=True)
    if args.resume_audio:
        report["completed"] = False
        ranked = [r for r in summary(report["trials"]) if r["model"] in report["finalists"] and r["successes"] >= 2][:3]
        for index in (1, 2):
            for row in ranked if index == 1 else reversed(ranked):
                report["trials"].append(trial(row["model"], "tts", PROMPTS[index], audio=True))
                save()
        report["completed"] = True
        report["completed_at"] = datetime.now().astimezone().isoformat()
        save()
        print("COMPLETE " + str(path), flush=True)
        return
    for model in MODELS:
        report["trials"].append(trial(model, "screen", PROMPTS[0]))
        save()
    successful = [r for r in report["summary"] if r["successes"]]
    finalists = [r["model"] for r in successful[:4]]
    if "gpt-5.6-terra" not in finalists and any(r["model"] == "gpt-5.6-terra" for r in successful):
        finalists.append("gpt-5.6-terra")
    report["finalists"] = finalists
    for index in (1, 2):
        order = finalists if index == 1 else list(reversed(finalists))
        for model in order:
            report["trials"].append(trial(model, f"repeat-{index}", PROMPTS[index]))
            save()
    ranked = [r for r in report["summary"] if r["model"] in finalists and r["successes"] >= 2]
    for row in ranked[:3]:
        report["trials"].append(trial(row["model"], "tts", PROMPTS[0], audio=True))
        save()
    report["completed"] = True
    report["completed_at"] = datetime.now().astimezone().isoformat()
    save()
    print("COMPLETE " + str(path), flush=True)


if __name__ == "__main__":
    main()
