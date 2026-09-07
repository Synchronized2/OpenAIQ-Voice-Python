from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from tts import EdgeSpeaker


class TtsPipelineTests(unittest.TestCase):
    def test_next_segment_is_synthesized_during_playback(self) -> None:
        events: dict[str, float] = {}

        def synthesize(text, path):
            events[f"synth-{text}-start"] = time.perf_counter()
            time.sleep(0.12)
            path.write_text(text, encoding="utf-8")
            events[f"synth-{text}-end"] = time.perf_counter()

        def play(path):
            text = path.read_text(encoding="utf-8")
            events[f"play-{text}-start"] = time.perf_counter()
            time.sleep(0.30)
            events[f"play-{text}-end"] = time.perf_counter()

        with (
            patch("tts.pygame.mixer.init"),
            patch("tts.pygame.mixer.music.stop"),
            patch("tts.pygame.mixer.quit"),
        ):
            speaker = EdgeSpeaker()
            speaker._synthesize = synthesize
            speaker._play = play
            try:
                speaker.enqueue("one")
                speaker.enqueue("two")
                speaker.wait()
            finally:
                speaker.close()

        self.assertLess(events["synth-two-start"], events["play-one-end"])
        self.assertGreaterEqual(events["play-two-start"], events["play-one-end"])


if __name__ == "__main__":
    unittest.main()
