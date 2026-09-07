from __future__ import annotations

import re
import time


def _normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]", text.casefold()))


def _canonical_syllable(value: str) -> str:
    value = value.casefold()
    for source, target in (("zh", "z"), ("ch", "c"), ("sh", "s")):
        if value.startswith(source):
            return target + value[len(source) :]
    return value


class WakeWordDetector:
    """Match configured literals and conservative Chinese homophones."""

    def __init__(
        self,
        phrases: tuple[str, ...],
        aliases: tuple[str, ...] = (),
        cooldown_seconds: float = 4.0,
        min_chars: int = 3,
        max_chars: int = 12,
    ) -> None:
        self.phrases = tuple(filter(None, (_normalize(value) for value in phrases)))
        self.aliases = tuple(filter(None, (_normalize(value) for value in aliases)))
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self._last_match_at = float("-inf")
        self._phonetic_targets = tuple(
            filter(None, (self._to_pinyin(value) for value in (*self.phrases, *self.aliases)))
        )

    def matches(self, text: str, now: float | None = None) -> bool:
        normalized = _normalize(text)
        if not self.min_chars <= len(normalized) <= self.max_chars:
            return False
        matched = any(value in normalized for value in (*self.phrases, *self.aliases))
        if not matched:
            candidate = self._to_pinyin(normalized)
            matched = any(self._contains_tokens(candidate, target) for target in self._phonetic_targets)
        if not matched:
            return False
        current = time.monotonic() if now is None else now
        if current - self._last_match_at < self.cooldown_seconds:
            return False
        self._last_match_at = current
        return True

    @staticmethod
    def _contains_tokens(candidate: tuple[str, ...], target: tuple[str, ...]) -> bool:
        if not target or len(candidate) < len(target):
            return False
        return any(
            candidate[index : index + len(target)] == target
            for index in range(len(candidate) - len(target) + 1)
        )

    @staticmethod
    def _to_pinyin(text: str) -> tuple[str, ...]:
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return ()
        return tuple(_canonical_syllable(value) for value in lazy_pinyin(text, errors="ignore"))

