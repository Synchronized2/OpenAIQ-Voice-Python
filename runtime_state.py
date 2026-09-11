from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeState(str, Enum):
    BOOTING = "booting"
    STANDBY = "standby"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    GENERATING = "generating"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RuntimeMachine:
    state: RuntimeState = RuntimeState.BOOTING
    active_worker: str | None = None

    @property
    def busy(self) -> bool:
        return self.active_worker is not None

    def is_running(self, worker: str) -> bool:
        return self.active_worker == worker

    def start(self, worker: str, state: RuntimeState) -> bool:
        if self.active_worker is not None:
            return False
        self.active_worker = worker
        self.state = state
        return True

    def transition(self, state: RuntimeState) -> None:
        self.state = state

    def finish(self, worker: str) -> bool:
        if self.active_worker != worker:
            return False
        self.active_worker = None
        return True
