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

    def test_each_queued_segment_keeps_its_voice_settings_snapshot(self) -> None:
        observed = []
        with (
            patch("tts.pygame.mixer.init"),
            patch("tts.pygame.mixer.music.stop"),
            patch("tts.pygame.mixer.quit"),
        ):
            speaker = EdgeSpeaker()

            def synthesize(text, path):
                observed.append(("synthesize", text, speaker._synth_options))
                path.write_text(text, encoding="utf-8")

            def play(_path):
                observed.append(("play", speaker._active_segment, speaker._play_options))

            speaker._synthesize = synthesize
            speaker._play = play
            try:
                speaker.configure(
                    voice="zh-CN-XiaoxiaoNeural", rate="-6%", volume=0.4
                )
                speaker.enqueue("one")
                speaker.configure(
                    voice="zh-CN-YunxiNeural", rate="+12%", volume=0.8
                )
                speaker.enqueue("two")
                speaker.wait()
            finally:
                speaker.close()

        synth = [item for item in observed if item[0] == "synthesize"]
        played = [item for item in observed if item[0] == "play"]
        self.assertEqual(
            [(item[1], item[2].voice, item[2].rate, item[2].volume) for item in synth],
            [
                ("one", "zh-CN-XiaoxiaoNeural", "-6%", 0.4),
                ("two", "zh-CN-YunxiNeural", "+12%", 0.8),
            ],
        )
        self.assertEqual(
            [(item[1], item[2].voice, item[2].rate, item[2].volume) for item in played],
            [
                ("one", "zh-CN-XiaoxiaoNeural", "-6%", 0.4),
                ("two", "zh-CN-YunxiNeural", "+12%", 0.8),
            ],
        )


if __name__ == "__main__":
    unittest.main()
