from __future__ import annotations

import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from chat import ChatSession, ModelResponseTimeout


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class ChatLatencyTests(unittest.TestCase):
    def setUp(self):
        self.chat = ChatSession.__new__(ChatSession)
        self.chat.messages = [{"role": "system", "content": "test"}]
        self.enterContext(patch("chat.lookup_weather", return_value=None))

    def test_heartbeats_do_not_extend_first_content_deadline(self):
        closed = []

        async def stream():
            try:
                while True:
                    yield chunk("")
                    await asyncio.sleep(0.01)
            finally:
                closed.append(True)

        self.chat._stream = stream
        started = time.monotonic()
        with patch("chat.config.CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 0.1), patch("chat.config.CHAT_TOTAL_TIMEOUT_SECONDS", 1):
            with self.assertRaisesRegex(ModelResponseTimeout, "没有返回正文"):
                self.chat.reply("hello")
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(closed, [True])
        self.assertEqual(len(self.chat.messages), 1)

    def test_total_deadline_stops_a_stream_after_first_content(self):
        closed = []

        async def stream():
            try:
                yield chunk("Hello")
                while True:
                    yield chunk("")
                    await asyncio.sleep(0.01)
            finally:
                closed.append(True)

        self.chat._stream = stream
        with patch("chat.config.CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 0.05), patch("chat.config.CHAT_TOTAL_TIMEOUT_SECONDS", 0.15):
            with self.assertRaisesRegex(ModelResponseTimeout, "模型生成超过"):
                self.chat.reply("hello")
        self.assertEqual(closed, [True])
        self.assertEqual(len(self.chat.messages), 1)

    def test_first_content_deadline_is_lifted_once_text_arrives(self):
        async def stream():
            yield chunk("Hello")
            await asyncio.sleep(0.12)
            yield chunk(" again")

        self.chat._stream = stream
        with patch("chat.config.CHAT_FIRST_TOKEN_TIMEOUT_SECONDS", 0.05), patch("chat.config.CHAT_TOTAL_TIMEOUT_SECONDS", 2):
            answer, cancelled = self.chat.reply("hello")
        self.assertEqual(answer, "Hello again")
        self.assertFalse(cancelled)
        self.assertGreater(self.chat.last_timings["total"], 0.1)
        self.assertLess(self.chat.last_timings["first_token"], self.chat.last_timings["total"])

    def test_reasoning_option_can_be_disabled_and_does_not_leak_to_other_models(self):
        for model, effort, expected in (("gpt-5.6-terra", "low", "low"), ("gpt-5.6-terra", "auto", None), ("qwen-test", "low", None)):
            with self.subTest(model=model, effort=effort):
                requests = []

                async def handle(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n")

                def factory(**kwargs):
                    return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))

                async def consume():
                    async for _ in self.chat._stream():
                        pass

                self.chat.api_key = "test-key"
                self.chat.model = model
                with patch("chat.AsyncOpenAI", side_effect=factory), patch("chat.config.CHAT_BASE_URL", "https://local-test.invalid/v1"), patch("chat.config.CHAT_REASONING_EFFORT", effort):
                    asyncio.run(consume())
                self.assertEqual(len(requests), 1)
                if expected is None:
                    self.assertNotIn("reasoning_effort", requests[0])
                else:
                    self.assertEqual(requests[0]["reasoning_effort"], expected)
