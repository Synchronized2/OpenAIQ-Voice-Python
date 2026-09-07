from __future__ import annotations

import unittest

from wake import WakeWordDetector


class WakeWordDetectorTests(unittest.TestCase):
    def make_detector(self) -> WakeWordDetector:
        return WakeWordDetector(
            ("孙悟空", "猴哥", "悟空", "大圣", "齐天大圣"),
            ("五空", "后哥", "大胜"),
            cooldown_seconds=4,
            min_chars=2,
            max_chars=12,
        )

    def test_literal_and_homophone_variants(self) -> None:
        for phrase in ("孙悟空", "猴哥", "悟空", "大圣", "齐天大圣"):
            with self.subTest(phrase=phrase):
                self.assertTrue(self.make_detector().matches(phrase, now=10))
        self.assertTrue(self.make_detector().matches("五空", now=10))
        self.assertTrue(self.make_detector().matches("大胜", now=10))

    def test_similar_everyday_phrase_does_not_wake(self) -> None:
        detector = self.make_detector()
        self.assertFalse(detector.matches("真空", now=10))
        self.assertFalse(detector.matches("谢谢观看", now=10))
        self.assertFalse(detector.matches("猴子", now=10))
        self.assertFalse(detector.matches("贾维斯", now=10))

    def test_long_background_sentence_is_filtered(self) -> None:
        detector = self.make_detector()
        self.assertFalse(detector.matches("电视里的人说齐天大圣然后继续播放节目", now=10))

    def test_cooldown_blocks_repeated_activation(self) -> None:
        detector = self.make_detector()
        self.assertTrue(detector.matches("悟空", now=10))
        self.assertFalse(detector.matches("猴哥", now=12))
        self.assertTrue(detector.matches("大圣", now=15))


if __name__ == "__main__":
    unittest.main()
