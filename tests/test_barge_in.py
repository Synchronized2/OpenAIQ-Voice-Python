from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tts import EdgeSpeaker


class BargeInTests(unittest.TestCase):
    def test_playing_event_only_covers_active_audio(self) -> None:
        started = threading.Event()
        segment_started = threading.Event()
        segments: list[str] = []
        release = threading.Event()

        def on_segment(text: str) -> None:
            segments.append(text)
            segment_started.set()

        class FakeMusic:
            @staticmethod
            def load(_path):
                pass

            @staticmethod
            def set_volume(_volume):
                pass

            @staticmethod
            def play():
                started.set()

            @staticmethod
            def get_busy():
                return not release.is_set()

            @staticmethod
            def unload():
                pass

            @staticmethod
            def stop():
                release.set()

        with (
            patch("tts.pygame.mixer.init"),
            patch("tts.pygame.mixer.music", FakeMusic),
            patch("tts.pygame.mixer.quit"),
        ):
            speaker = EdgeSpeaker(on_segment_start=on_segment)
            try:
                speaker._active_segment = "当前朗读句子"
                player = threading.Thread(target=speaker._play, args=(Path("unused.mp3"),))
                player.start()
                self.assertTrue(started.wait(1.0))
                self.assertTrue(speaker.playing.wait(1.0))
                self.assertTrue(segment_started.wait(1.0))
                self.assertEqual(segments, ["当前朗读句子"])
                release.set()
                player.join(1.0)
                self.assertFalse(speaker.playing.is_set())
            finally:
                release.set()
                speaker.close()

    def test_wait_until_playing_can_be_cancelled(self) -> None:
        with (
            patch("tts.pygame.mixer.init"),
            patch("tts.pygame.mixer.music.stop"),
            patch("tts.pygame.mixer.quit"),
        ):
            speaker = EdgeSpeaker()
            stopped = threading.Event()
            try:
                stopped.set()
                started = time.perf_counter()
                self.assertFalse(speaker.wait_until_playing(stopped))
                self.assertLess(time.perf_counter() - started, 0.2)
            finally:
                speaker.close()


if __name__ == "__main__":
    unittest.main()
