"""Measure first text and sustained streaming, keeping TTS/app settings intact."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import config
from benchmark_models import MODELS
from chat import SentenceSegmenter


PROMPTS = (
    "请用约250个汉字、三段连贯自然的文字，给久坐上班族介绍下班后的放松方法。每段两三句。不要标题、列表、代码或思考过程，也不要询问我。",
    "请用约250个汉字、三段连贯自然的文字，介绍怎样度过轻松又充实的周末。每段两三句。不要标题、列表、代码或思考过程，也不要询问我。",
)


async def trial(model: str, prompt: str, round_index: int, include_usage: bool = True) -> dict:
    result = {"model": model, "round": round_index, "include_usage": include_usage}
    effort = config.CHAT_REASONING_EFFORT if model.startswith("gpt-5") else "auto"
    result["effort"] = effort
    options = {"reasoning_effort": effort} if effort != "auto" else {}
    if include_usage:
        options["stream_options"] = {"include_usage": True}
    parts, packets = [], []
    usage = None
    started = time.perf_counter()
    segmenter = SentenceSegmenter(config.TTS_SEGMENT_MAX_CHARS, config.TTS_SEGMENT_MIN_CHARS)

    def elapsed():
        return time.perf_counter() - started

    async def headers(response):
        result["headers_s"] = round(elapsed(), 3)
        result["http_status"] = response.status_code

    print(f"START round={round_index} model={model}", flush=True)
    try:
        async with asyncio.timeout(50):
            async with asyncio.timeout(20) as first_content_deadline:
                async with AsyncOpenAI(
                    api_key=config.chat_api_key(), base_url=config.CHAT_BASE_URL,
                    max_retries=0, timeout=httpx.Timeout(30, connect=10),
                    http_client=httpx.AsyncClient(event_hooks={"response": [headers]}),
                ) as client:
                    stream = await client.chat.completions.create(
                        model=model, messages=[{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                        temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS, stream=True, **options,
                    )
                    async with stream:
                        async for chunk in stream:
                            if chunk.usage:
                                usage = chunk.usage.model_dump()
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            if choice.finish_reason:
                                result["finish_reason"] = choice.finish_reason
                            text = choice.delta.content
                            if not isinstance(text, str) or not text:
                                continue
                            if not parts:
                                first_content_deadline.reschedule(None)
                            parts.append(text)
                            packets.append({"seconds": elapsed(), "chars": len(text)})
                            sentences = segmenter.feed(text)
                            if sentences and "first_sentence_s" not in result:
                                result["first_sentence_s"] = round(elapsed(), 3)
        if segmenter.flush() and "first_sentence_s" not in result:
            result["first_sentence_s"] = round(elapsed(), 3)
        if not parts:
            raise RuntimeError("Empty response")
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).replace(config.chat_api_key(), "[REDACTED]")[:500]
        if include_usage and getattr(exc, "status_code", None) == 400 and "stream_options" in str(exc):
            return await trial(model, prompt, round_index, include_usage=False)
    result["request_total_s"] = round(elapsed(), 3)
    result["reply"] = "".join(parts)
    result["text_chars"] = len(result["reply"])
    result["usage"] = usage
    result["packets"] = packets
    if packets:
        first, last = packets[0]["seconds"], packets[-1]["seconds"]
        duration = last - first
        result["first_token_s"] = round(first, 3)
        result["last_text_s"] = round(last, 3)
        result["generation_span_s"] = round(duration, 3)
        result["burst_output"] = duration < 0.1
        result["chars_per_second"] = round((result["text_chars"] - packets[0]["chars"]) / duration, 2) if duration >= 0.1 else None
        result["end_to_end_chars_per_second"] = round(result["text_chars"] / last, 2)
        details = (usage or {}).get("completion_tokens_details") or {}
        tokens = (usage or {}).get("completion_tokens")
        reasoning = details.get("reasoning_tokens")
        result["reported_completion_tokens_per_second"] = round(tokens / duration, 2) if tokens is not None and duration >= 0.1 else None
        if tokens is not None and reasoning is not None:
            result["visible_tokens_per_second_estimate"] = round((tokens - reasoning) / duration, 2) if duration >= 0.1 else None
        gaps = [b["seconds"] - a["seconds"] for a, b in zip(packets, packets[1:])]
        result["max_interpacket_gap_s"] = round(max(gaps, default=0), 3)
    print(json.dumps({k: v for k, v in result.items() if k not in {"reply", "packets", "usage", "error"}}, ensure_ascii=True), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 3:
        parser.error("rounds must be between 1 and 3")
    folder = Path(__file__).resolve().parent / "logs"
    folder.mkdir(exist_ok=True)
    path = folder / f"stream-benchmark-{datetime.now():%Y%m%d-%H%M%S}.json"
    report = {"started_at": datetime.now().astimezone().isoformat(), "completed": False,
              "method": "Sequential fresh conversation/client; same prompt per round; current app temperature, max_tokens and reasoning policy; Edge TTS unchanged.",
              "rate_note": "Character rate excludes first packet. API completion token rate may include reasoning unless the node supplies completion_tokens_details. SSE packets are not tokens.",
              "prompts": list(PROMPTS), "trials": []}
    print(f"REPORT {path}", flush=True)
    for index in range(args.rounds):
        order = args.models if index % 2 == 0 else reversed(args.models)
        for model in order:
            report["trials"].append(await trial(model, PROMPTS[index % len(PROMPTS)], index + 1))
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["completed"] = True
    report["completed_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE " + str(path), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
