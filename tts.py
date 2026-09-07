from __future__ import annotations

import queue
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import edge_tts
import pygame
from edge_tts.exceptions import NoAudioReceived

import config
from cancellation import OperationCancelled, run_cancellable
from diagnostics import log


class EdgeSpeaker:
    def __init__(self, on_segment_start: Callable[[str], None] | None = None) -> None:
        pygame.mixer.init()
        self._text_queue: queue.Queue[tuple[str, threading.Event] | None] = queue.Queue()
        self._audio_queue: queue.Queue[tuple[Path, str, threading.Event] | None] = queue.Queue()
        self._turn_lock = threading.RLock()
        self._on_segment_start = on_segment_start
        self._active_segment = ""
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._synth_cancel = self._cancelled
        self._play_cancel = self._cancelled
        self._failed = threading.Event()
        self.playing = threading.Event()
        self._closed = False
        self._synth_worker = threading.Thread(
            target=self._run_synthesizer,
            name="edge-tts-synthesizer",
            daemon=True,
        )
        self._play_worker = threading.Thread(
            target=self._run_player,
            name="edge-tts-player",
            daemon=True,
        )
        self._synth_worker.start()
        self._play_worker.start()

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        self._text_queue.put(None)
        self._wait_queue(self._text_queue, timeout=5.0)
        self._synth_worker.join(timeout=5)
        self._audio_queue.put(None)
        self._wait_queue(self._audio_queue, timeout=5.0)
        self._play_worker.join(timeout=5)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        self._closed = True

    def stop(self) -> None:
        with self._turn_lock:
            self._cancelled.set()
            log.info("tts.cancel generation=%s", id(self._cancelled))
            pygame.mixer.music.stop()
            self._drain_text_queue()
            self._drain_audio_queue()

    def begin_response(self) -> None:
        with self._turn_lock:
            self.stop()
            with self._error_lock:
                self._error = None
            self._failed.clear()
            self._cancelled = threading.Event()

    def enqueue(self, text: str) -> None:
        text = text.strip()
        with self._turn_lock:
            if text and not self._cancelled.is_set() and not self._closed:
                self._text_queue.put((text, self._cancelled))

    def wait(self) -> None:
        # Text must first become audio before the audio queue can be considered complete.
        self._wait_queue(self._text_queue)
        self._wait_queue(self._audio_queue)
        with self._error_lock:
            error = self._error
        if error:
            raise error

    def wait_interruptible(self, interrupted: threading.Event) -> bool:
        while not interrupted.wait(0.05):
            if self._queues_idle():
                self.wait()
                return True
        return False

    def wait_until_playing(self, stopped: threading.Event) -> bool:
        """Wait until audio playback starts, or until the caller cancels."""
        while not stopped.wait(0.05):
            if self.playing.is_set():
                return True
            if self._failed.is_set():
                return False
        return False

    def speak(self, text: str) -> None:
        self.begin_response()
        self.enqueue(text)
        self.wait()

    def _run_synthesizer(self) -> None:
        while True:
            item = self._text_queue.get()
            cancelled = None
            try:
                if item is None:
                    return
                text, cancelled = item
                self._synth_cancel = cancelled
                if cancelled.is_set() or self._failed.is_set():
                    continue
                path = self._new_temp_path()
                try:
                    self._synthesize(text, path)
                    with self._turn_lock:
                        if cancelled.is_set() or self._failed.is_set():
                            path.unlink(missing_ok=True)
                        else:
                            self._audio_queue.put((path, text, cancelled))
                except Exception:
                    path.unlink(missing_ok=True)
                    raise
            except OperationCancelled:
                pass
            except Exception as exc:
                with self._turn_lock:
                    if cancelled is self._cancelled and not cancelled.is_set():
                        self._set_error(exc)
            finally:
                self._text_queue.task_done()

    def _run_player(self) -> None:
        while True:
            item = self._audio_queue.get()
            path = item[0] if item is not None else None
            try:
                if item is None:
                    return
                path, self._active_segment, self._play_cancel = item
                if not self._play_cancel.is_set() and not self._failed.is_set():
                    self._play(path)
            except Exception as exc:
                with self._turn_lock:
                    if self._play_cancel is self._cancelled and not self._play_cancel.is_set():
                        self._set_error(exc)
            finally:
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except PermissionError:
                        pass
                self._active_segment = ""
                self._audio_queue.task_done()

    def _play(self, path: Path) -> None:
        cancelled = self._play_cancel
        with self._turn_lock:
            if cancelled.is_set():
                return
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            log.info("tts.play generation=%s chars=%s", id(cancelled), len(self._active_segment))
            self.playing.set()
            if self._on_segment_start and self._active_segment:
                self._on_segment_start(self._active_segment)
        try:
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy() and not cancelled.is_set():
                clock.tick(50)
        finally:
            self.playing.clear()
            pygame.mixer.music.unload()
            if self._on_segment_start:
                self._on_segment_start("")

    def _set_error(self, error: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error
        self._failed.set()

    def _queues_idle(self) -> bool:
        with self._text_queue.all_tasks_done:
            text_idle = self._text_queue.unfinished_tasks == 0
        with self._audio_queue.all_tasks_done:
            audio_idle = self._audio_queue.unfinished_tasks == 0
        return text_idle and audio_idle

    @staticmethod
    def _wait_queue(work_queue: queue.Queue, timeout: float | None = None) -> bool:
        """Wait in interruptible intervals instead of Queue.join's indefinite wait."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            with work_queue.all_tasks_done:
                if work_queue.unfinished_tasks == 0:
                    return True
                interval = 0.1
                if deadline is not None:
                    interval = min(interval, max(0.0, deadline - time.monotonic()))
                    if interval == 0.0:
                        return False
                work_queue.all_tasks_done.wait(timeout=interval)

    def _drain_text_queue(self) -> None:
        while True:
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._text_queue.task_done()

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                item = self._audio_queue.get_nowait()
            except queue.Empty:
                return
            else:
                if item is not None:
                    path, _text, _cancelled = item
                    path.unlink(missing_ok=True)
                self._audio_queue.task_done()

    @staticmethod
    def _new_temp_path() -> Path:
        handle = tempfile.NamedTemporaryFile(prefix="openaiq-voice-", suffix=".mp3", delete=False)
        path = Path(handle.name)
        handle.close()
        return path

    def _synthesize(self, text: str, path: Path) -> None:
        cancelled = self._synth_cancel
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, config.TTS_RETRIES + 1):
            path.unlink(missing_ok=True)
            try:
                communicate = edge_tts.Communicate(
                    text,
                    config.TTS_VOICE,
                    rate=config.TTS_RATE,
                    connect_timeout=10,
                    receive_timeout=60,
                )
                run_cancellable(communicate.save(str(path)), cancelled.is_set)
                if path.exists() and path.stat().st_size > 0:
                    log.info("tts.synthesize.end generation=%s seconds=%.3f", id(cancelled), time.perf_counter() - started)
                    return
                raise NoAudioReceived("Edge TTS returned an empty audio file")
            except NoAudioReceived as exc:
                last_error = exc
                if attempt < config.TTS_RETRIES:
                    print(f"[TTS] 未收到音频，正在重试 ({attempt}/{config.TTS_RETRIES})…")
                    if cancelled.wait(config.TTS_RETRY_DELAY_SECONDS * attempt):
                        raise OperationCancelled()
        raise RuntimeError(
            f"Edge TTS 连续 {config.TTS_RETRIES} 次未返回音频，请稍后重试或检查网络代理。"
        ) from last_error
