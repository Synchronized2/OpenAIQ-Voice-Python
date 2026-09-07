from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable

import httpx
from openai import AsyncOpenAI

import config
from cancellation import OperationCancelled, run_cancellable
from diagnostics import log
from weather import is_weather_query, lookup_weather


SENTENCE_END = re.compile(r"[。！？!?；;\n]+[”’」』）)]*")
SOFT_BREAKS = "，、,:： "


class ModelResponseTimeout(TimeoutError):
    """The model exceeded the content deadline, even if SSE heartbeats arrived."""


class SentenceSegmenter:
    def __init__(self, max_chars: int, min_chars: int = 1) -> None:
        self.max_chars = max_chars
        self.min_chars = min(min_chars, max_chars)
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        segments: list[str] = []
        while self.buffer:
            ending = next(
                (match for match in SENTENCE_END.finditer(self.buffer) if match.end() >= self.min_chars),
                None,
            )
            if ending:
                end = ending.end()
            elif len(self.buffer) >= self.max_chars:
                end = self._soft_break()
            else:
                break
            segment = self.buffer[:end].strip()
            self.buffer = self.buffer[end:]
            if segment:
                segments.append(segment)
        return segments

    def flush(self) -> str:
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining

    def _soft_break(self) -> int:
        window = self.buffer[: self.max_chars]
        candidates = [window.rfind(mark) + 1 for mark in SOFT_BREAKS]
        best = max(candidates)
        return best if best >= self.max_chars // 2 else self.max_chars


class ChatSession:
    def __init__(self, model: str | None = None) -> None:
        api_key = config.chat_api_key()
        if not api_key:
            raise RuntimeError(
                "未配置聊天 Key：请填写 config.py 的 CHAT_API_KEY，"
                "或设置系统环境变量 PCIE_API_KEY / OPENAI_API_KEY。"
            )
        self.api_key = api_key
        self.model = model or config.CHAT_MODEL
        self.messages: list[dict[str, str]] = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        self.last_timings: dict[str, float | None] = {}

    async def _stream(self):
        started = time.perf_counter()
        effort = str(getattr(config, "CHAT_REASONING_EFFORT", "auto")).lower()
        options = {}
        if self.model.startswith("gpt-5") and effort != "auto":
            if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("CHAT_REASONING_EFFORT 配置无效；使用 low 或 auto。")
            options["reasoning_effort"] = effort
        log.info("llm.request_start model=%s effort=%s", self.model, options.get("reasoning_effort", "auto"))
        async with AsyncOpenAI(
            api_key=self.api_key, base_url=config.CHAT_BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0), max_retries=0,
        ) as client:
            log.info("llm.client_ready seconds=%.3f", time.perf_counter() - started)
            stream = await client.chat.completions.create(
                model=self.model, messages=self.messages,
                temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
                stream=True, **options,
            )
            log.info("llm.headers seconds=%.3f", time.perf_counter() - started)
            first_event = True
            async with stream:
                async for chunk in stream:
                    if first_event:
                        log.info("llm.first_event seconds=%.3f", time.perf_counter() - started)
                        first_event = False
                    yield chunk

    def reply(
        self,
        user_text: str,
        on_sentence: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        stopped = should_stop or (lambda: False)
        self.last_timings = {}
        if stopped():
            return "", True
        if is_weather_query(user_text):
            print("[天气] 正在查询实时天气…", flush=True)
        weather_context = lookup_weather(user_text)
        if stopped():
            return "", True
        content = user_text
        if weather_context:
            content = (
                f"{user_text}\n\n[实时工具上下文]\n{weather_context}"
            )
        user_message = {"role": "user", "content": content}
        self.messages.append(user_message)
        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        interrupted = False
        segmenter = SentenceSegmenter(
            config.TTS_SEGMENT_MAX_CHARS,
            config.TTS_SEGMENT_MIN_CHARS,
        )
        print("\nAI：", end="", flush=True)

        async def consume(first_content_deadline):
            nonlocal first_token_at
            stream = self._stream()
            try:
                async for chunk in stream:
                    if stopped():
                        raise OperationCancelled()
                    text = chunk.choices[0].delta.content if chunk.choices else None
                    if not text:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        first_content_deadline.reschedule(None)
                        log.info("llm.first_token seconds=%.3f", first_token_at - started)
                    parts.append(text)
                    print(text, end="", flush=True)
                    if on_text:
                        on_text(text)
                    if on_sentence:
                        for sentence in segmenter.feed(text):
                            on_sentence(sentence)
            finally:
                await stream.aclose()

        async def consume_with_deadlines():
            first_timeout = float(getattr(config, "CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 30.0))
            total_timeout = float(getattr(config, "CHAT_TOTAL_TIMEOUT_SECONDS", 90.0))
            try:
                async with asyncio.timeout(total_timeout):
                    async with asyncio.timeout(first_timeout) as deadline:
                        await consume(deadline)
            except TimeoutError as exc:
                if first_token_at is None:
                    raise ModelResponseTimeout(
                        f"模型在 {first_timeout:g} 秒内没有返回正文。中转服务可能正在排队或推理；"
                        "请重试或切换模型。这不是 TTS 故障。"
                    ) from exc
                raise ModelResponseTimeout(f"模型生成超过 {total_timeout:g} 秒，已停止本轮请求。") from exc

        try:
            run_cancellable(consume_with_deadlines(), stopped)
        except OperationCancelled:
            interrupted = True
        except BaseException as exc:
            log.info("llm.error kind=%s seconds=%.3f", type(exc).__name__, time.perf_counter() - started)
            self._rollback_message(user_message)
            raise

        answer = "".join(parts).strip()
        if on_sentence and not interrupted:
            remaining = segmenter.flush()
            if remaining:
                on_sentence(remaining)
        total = time.perf_counter() - started
        self.last_timings = {
            "first_token": first_token_at - started if first_token_at is not None else None,
            "total": total,
        }
        log.info("llm.end seconds=%.3f cancelled=%s", total, interrupted)
        first = (first_token_at - started) if first_token_at else total
        print(f"\n[耗时] 首字 {first:.2f}s / 总计 {total:.2f}s")
        if not answer and not interrupted:
            self._rollback_message(user_message)
            raise RuntimeError("聊天模型返回为空。")
        if interrupted:
            self._rollback_message(user_message)
        elif answer:
            self.messages.append({"role": "assistant", "content": answer})
            self._trim_history()
        return answer, interrupted

    def reset(self) -> None:
        self.messages = [self.messages[0]]

    def _rollback_message(self, user_message: dict[str, str]) -> None:
        """Remove a user turn that did not complete successfully."""
        if self.messages and self.messages[-1] is user_message:
            self.messages.pop()

    def _trim_history(self) -> None:
        limit = config.HISTORY_TURNS * 2
        if len(self.messages) > limit + 1:
            self.messages = [self.messages[0], *self.messages[-limit:]]
