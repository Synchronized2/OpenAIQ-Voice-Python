from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class BargeSession:
    """One monitor owns its cancellation, result and consumption lifetime."""

    stopped: threading.Event = field(default_factory=threading.Event)
    triggered: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    text: str = ""
    consumed: bool = False

    def take_result(self) -> tuple[bool, str]:
        if not self.done.is_set() or self.consumed:
            return False, ""
        self.consumed = True
        if self.stopped.is_set():
            return False, ""
        return self.triggered.is_set(), self.text
