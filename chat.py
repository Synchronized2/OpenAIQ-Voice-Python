from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable

import httpx
from openai import AsyncOpenAI

import config
from cancellation import OperationCancelled, run_cancellable
from diagnostics import log
from weather import is_weather_query, lookup_weather
from image_generation import IMAGE_GENERATION_TOOL, IMAGE_TOOL_GUIDANCE


SENTENCE_END = re.compile(r"[。！？!?；;\n]+[”’」』）)]*")
SOFT_BREAKS = "，、,:： "
IMAGE_CREATION_VERB = re.compile(
    r"生成|绘制|画(?:一|个|张|幅)?|制作|创建|设计|做(?:一|个|张|幅)?|来(?:一|个|张|幅)?|"
    r"generate|create|draw|paint|make",
    re.IGNORECASE,
)
IMAGE_CREATION_OBJECT = re.compile(
    r"图片|照片|图像|插画|海报|壁纸|头像|画作|图|"
    r"image|photo|picture|illustration|poster|wallpaper|avatar",
    re.IGNORECASE,
)


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
    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        api_key = (api_key if api_key is not None else config.chat_api_key()).strip()
        if not api_key:
            raise RuntimeError(
                "未配置聊天 Key：请填写 config.py 的 CHAT_API_KEY，"
                "或设置系统环境变量 PCIE_API_KEY / OPENAI_API_KEY。"
            )
        self.api_key = api_key
        self.base_url = (
            base_url if base_url is not None else config.CHAT_BASE_URL
        ).strip()
        if not self.base_url:
            raise RuntimeError("未配置模型服务 URL。")
        self.model = model or config.CHAT_MODEL
        self.messages: list[dict[str, str]] = [{
            "role": "system",
            "content": f"{config.SYSTEM_PROMPT}\n\n{IMAGE_TOOL_GUIDANCE}",
        }]
        self.last_timings: dict[str, float | None] = {}
        self.last_tool_call: dict[str, str] | None = None

    def _needs_image_tool_preflight(self, user_text: str) -> bool:
        incompatible = tuple(
            str(item).casefold()
            for item in getattr(
                config, "IMAGE_TOOL_INCOMPATIBLE_MODELS", ("gpt-5.3-codex-spark",)
            )
        )
        model = str(getattr(self, "model", "")).casefold()
        return (
            any(marker and marker in model for marker in incompatible)
            and IMAGE_CREATION_VERB.search(user_text) is not None
            and IMAGE_CREATION_OBJECT.search(user_text) is not None
        )

    async def _route_image_tool(self, user_text: str) -> dict[str, str] | None:
        router_model = str(
            getattr(config, "IMAGE_TOOL_ROUTER_MODEL", "gpt-5.6-luna")
        ).strip()
        if not router_model or router_model.casefold() == self.model.casefold():
            return None
        timeout = float(getattr(config, "IMAGE_TOOL_ROUTER_TIMEOUT_SECONDS", 15.0))
        log.info(
            "llm.image_router_start selected_model=%s router_model=%s",
            self.model,
            router_model,
        )
        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            max_retries=0,
        ) as client:
            async with asyncio.timeout(timeout):
                response = await client.chat.completions.create(
                    model=router_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{IMAGE_TOOL_GUIDANCE}\n"
                                "你只负责判断当前用户消息是否需要创建新图片。"
                                "需要时调用 generate_image；不需要时只回答 NO_IMAGE。"
                            ),
                        },
                        {"role": "user", "content": user_text},
                    ],
                    tools=[IMAGE_GENERATION_TOOL],
                    tool_choice="auto",
                    max_tokens=300,
                )
        choices = getattr(response, "choices", None) or ()
        message = getattr(choices[0], "message", None) if choices else None
        calls = getattr(message, "tool_calls", None) or ()
        if not calls:
            log.info("llm.image_router_end tool_call=false")
            return None
        function = getattr(calls[0], "function", None)
        result = self._parse_image_tool_call(
            str(getattr(function, "name", "") or ""),
            str(getattr(function, "arguments", "") or "{}"),
            str(getattr(calls[0], "id", "") or ""),
        )
        log.info("llm.image_router_end tool_call=true")
        return result

    def _preflight_image_tool(
        self, user_text: str, stopped: Callable[[], bool]
    ) -> dict[str, str] | None:
        if not self._needs_image_tool_preflight(user_text):
            return None
        try:
            return run_cancellable(self._route_image_tool(user_text), stopped)
        except OperationCancelled:
            raise
        except BaseException as exc:
            log.info("llm.image_router_error kind=%s", type(exc).__name__)
            return None

    @staticmethod
    def _parse_image_tool_call(
        name: str, arguments: str, call_id: str = ""
    ) -> dict[str, str]:
        if name != "generate_image":
            raise RuntimeError(f"对话模型请求了未知工具：{name or '未命名'}")
        try:
            decoded = json.loads(arguments or "{}")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("对话模型返回了无效的生图工具参数。") from exc
        prompt = decoded.get("prompt") if isinstance(decoded, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("对话模型没有提供有效的生图提示词。")
        return {"id": call_id, "name": name, "prompt": prompt.strip()}

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
            api_key=self.api_key,
            base_url=getattr(self, "base_url", config.CHAT_BASE_URL),
            timeout=httpx.Timeout(30.0, connect=10.0), max_retries=0,
        ) as client:
            log.info("llm.client_ready seconds=%.3f", time.perf_counter() - started)
            stream = await client.chat.completions.create(
                model=self.model, messages=self.messages,
                temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
                stream=True,
                tools=[IMAGE_GENERATION_TOOL],
                tool_choice="auto",
                **options,
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
        self.last_tool_call = None
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
        try:
            routed_tool_call = self._preflight_image_tool(user_text, stopped)
        except OperationCancelled:
            self._rollback_message(user_message)
            return "", True
        if routed_tool_call is not None:
            total = time.perf_counter() - started
            self.last_tool_call = routed_tool_call
            self.last_timings = {"first_token": total, "total": total}
            log.info("llm.end seconds=%.3f cancelled=false routed_tool=true", total)
            print(f"\nAI：\n[耗时] 工具判定 {total:.2f}s")
            return "", False
        first_token_at: float | None = None
        parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
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
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue
                    for tool_call in getattr(delta, "tool_calls", None) or ():
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            first_content_deadline.reschedule(None)
                            log.info("llm.first_tool_call seconds=%.3f", first_token_at - started)
                        index = int(getattr(tool_call, "index", 0) or 0)
                        entry = tool_parts.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if getattr(tool_call, "id", None):
                            entry["id"] = str(tool_call.id)
                        function = getattr(tool_call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                entry["name"] += str(function.name)
                            if getattr(function, "arguments", None):
                                entry["arguments"] += str(function.arguments)
                    text = getattr(delta, "content", None)
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
        if tool_parts and not interrupted:
            call = tool_parts[min(tool_parts)]
            try:
                self.last_tool_call = self._parse_image_tool_call(
                    call["name"], call["arguments"], call["id"]
                )
            except RuntimeError:
                self._rollback_message(user_message)
                raise
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
        if not answer and not interrupted and self.last_tool_call is None:
            self._rollback_message(user_message)
            raise RuntimeError("聊天模型返回为空。")
        if interrupted:
            self._rollback_message(user_message)
        elif self.last_tool_call is not None:
            # Keep the user turn. The GUI records a natural assistant result after
            # the image tool finishes, avoiding an unresolved tool-call message.
            pass
        elif answer:
            self.messages.append({"role": "assistant", "content": answer})
            self._trim_history()
        return answer, interrupted

    def record_image_result(self, prompt: str, path: str | None) -> None:
        if path:
            content = f"已按要求生成图片并保存到本地：{path}"
        else:
            content = f"尝试生成图片但没有成功。提示词：{prompt}"
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

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
