from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chat import ChatSession


class FakeStream:
    def __init__(self) -> None:
        self.produced = 0
        self.closed = False

    async def __aiter__(self):
        for text in ("第一句已经足够长，可以立即进入语音合成队列。", "第二句不应被处理。"):
            self.produced += 1
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )

    async def aclose(self) -> None:
        self.closed = True


class ChatInterruptTests(unittest.TestCase):
    def test_interrupt_closes_stream_and_drops_later_text(self) -> None:
        stream = FakeStream()
        session = ChatSession.__new__(ChatSession)
        session._stream = lambda: stream
        session.messages = [{"role": "system", "content": "test"}]
        spoken: list[str] = []
        streamed: list[str] = []

        with patch("chat.config.TTS_SEGMENT_MIN_CHARS", 1):
            answer, interrupted = session.reply(
                "test",
                on_sentence=spoken.append,
                on_text=streamed.append,
                should_stop=lambda: stream.produced > 1,
            )

        self.assertTrue(interrupted)
        self.assertTrue(stream.closed)
        self.assertIn("第一句", answer)
        self.assertNotIn("第二句", answer)
        self.assertEqual(len(spoken), 1)
        self.assertEqual(streamed, ["第一句已经足够长，可以立即进入语音合成队列。"])
        self.assertEqual(session.messages, [{"role": "system", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
