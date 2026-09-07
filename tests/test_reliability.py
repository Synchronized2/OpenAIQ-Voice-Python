from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import httpx
from openai import AsyncOpenAI
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from agent import LocalAgent
from asr import StreamingASR, CAPTURE_BLOCK_SAMPLES
from chat import ChatSession
from gui import AiCoreWidget, AssistantCore, VoiceWindow, WaveformWidget, load_ui_preferences
from runtime_state import RuntimeMachine, RuntimeState
from tts import EdgeSpeaker
from voice_session import BargeSession


class CancellationTests(unittest.TestCase):
    def _cancel_real_client(self, stall_headers):
        entered, closed, stopped = threading.Event(), threading.Event(), threading.Event()
        results, failures = [], []

        class StalledBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
                entered.set()
                await asyncio.sleep(60)

            async def aclose(self):
                closed.set()

        async def handle(request):
            if stall_headers:
                try:
                    entered.set()
                    await asyncio.sleep(60)
                finally:
                    closed.set()
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=StalledBody())

        def client_factory(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))

        def request():
            try:
                results.append(chat.reply("test", should_stop=stopped.is_set))
            except BaseException as exc:
                failures.append(exc)

        with patch("chat.config.chat_api_key", return_value="test-key"), patch("chat.config.CHAT_BASE_URL", "https://local-test.invalid/v1"), patch("chat.AsyncOpenAI", side_effect=client_factory), patch("chat.lookup_weather", return_value=None):
            chat = ChatSession(model="test-model")
            worker = threading.Thread(target=request)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                stopped.set()
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(failures, [])
                self.assertTrue(closed.is_set())
                self.assertEqual(results, [("" if stall_headers else "Hello", True)])
                self.assertEqual(len(chat.messages), 1)
            finally:
                stopped.set()
                worker.join(2)

    def test_real_openai_client_cancels_pending_headers(self):
        self._cancel_real_client(stall_headers=True)

    def test_real_openai_client_cancels_stalled_body(self):
        self._cancel_real_client(stall_headers=False)

    def test_cancel_while_llm_waits_for_network(self):
        entered, closed, stopped = threading.Event(), threading.Event(), threading.Event()
        async def stream():
            try:
                entered.set()
                await asyncio.sleep(60)
                yield None
            finally:
                closed.set()
        chat = ChatSession.__new__(ChatSession)
        chat.messages = [{"role": "system", "content": "test"}]
        chat._stream = stream
        results = []
        with patch("chat.lookup_weather", return_value=None):
            worker = threading.Thread(target=lambda: results.append(chat.reply("test", should_stop=stopped.is_set)))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                stopped.set()
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(closed.is_set())
                self.assertEqual(results, [("", True)])
                self.assertEqual(len(chat.messages), 1)
            finally:
                stopped.set()
                worker.join(2)

    def test_new_tts_cancels_inflight_synthesis_and_drops_old_audio(self):
        entered, closed = threading.Event(), threading.Event()
        played = []
        class Communicate:
            def __init__(self, text, *args, **kwargs):
                self.text = text
            async def save(self, path):
                if self.text == "old":
                    try:
                        entered.set()
                        await asyncio.sleep(60)
                    finally:
                        closed.set()
                else:
                    Path(path).write_bytes(b"audio")

        with patch("tts.pygame.mixer.init"), patch("tts.pygame.mixer.music.stop"), patch("tts.pygame.mixer.quit"), patch("tts.edge_tts.Communicate", Communicate):
            speaker = EdgeSpeaker()
            speaker._play = lambda path: played.append(speaker._active_segment)
            try:
                speaker.enqueue("old")
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                speaker.begin_response()
                self.assertLess(time.monotonic() - started, 0.5)
                speaker.enqueue("new")
                self.assertTrue(closed.wait(2))
                self.assertTrue(speaker._wait_queue(speaker._text_queue, timeout=2))
                self.assertTrue(speaker._wait_queue(speaker._audio_queue, timeout=2))
                self.assertEqual(played, ["new"])
            finally:
                speaker.close()

    def test_asr_cancel_during_transcription_returns_no_text(self):
        stopped = threading.Event()
        asr = StreamingASR.__new__(StreamingASR)
        class InputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
            def __enter__(self):
                for _ in range(3):
                    self.callback(np.zeros((CAPTURE_BLOCK_SAMPLES, 1), dtype=np.float32), 0, None, None)
                return self
            def __exit__(self, *args):
                pass
        asr._sd = SimpleNamespace(InputStream=InputStream)
        asr._vad = SimpleNamespace(reset=lambda: None, detect_chunk=lambda audio: [SimpleNamespace(is_speech_start=True, is_speech_end=True)])
        asr.device = None
        asr.pre_roll_samples = 6400
        asr.post_roll_samples = 4160
        asr.vad_end_silence = 0.7
        asr.debug_audio = False
        def transcribe(audio):
            stopped.set()
            return "open application"
        asr.transcribe = transcribe
        self.assertEqual(asr.listen_once(stop_event=stopped), "")


class GuiReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_monitor_completion_cannot_consume_barge_result(self):
        session = BargeSession()
        session.text = "next question"
        session.triggered.set()
        session.done.set()
        runtime = RuntimeMachine()
        runtime.start("chat", RuntimeState.PROCESSING)
        controller = SimpleNamespace(barge_session=session, exiting=False, runtime=runtime)
        VoiceWindow._barge_monitor_finished(controller, session)
        self.assertEqual(VoiceWindow._finish_barge_in_monitor(controller, session), (True, "next question"))
        self.assertEqual(VoiceWindow._finish_barge_in_monitor(controller, session), (False, ""))

    def test_cancelled_and_superseded_asr_cannot_submit_actions(self):
        old, current = threading.Event(), threading.Event()
        controller = SimpleNamespace(asr_stop=current, exiting=False, _queue_prompt=Mock())
        VoiceWindow._asr_finished(controller, "open application", False, old)
        current.set()
        VoiceWindow._asr_finished(controller, "open application", False, current)
        controller._queue_prompt.assert_not_called()

    def test_cancelled_barge_result_cannot_submit(self):
        session = BargeSession()
        session.stopped.set()
        controller = SimpleNamespace(barge_session=session, exiting=False, _queue_prompt=Mock())
        VoiceWindow._barge_in_finished(controller, "next question", session)
        controller._queue_prompt.assert_not_called()

    def test_early_barge_ui_completion_submits_next_question_once(self):
        recognized, release_chat = threading.Event(), threading.Event()

        def listen_once(**kwargs):
            kwargs["on_speech_start"]()
            recognized.set()
            return "next question"

        def reply(*args, **kwargs):
            if not release_chat.wait(3):
                raise TimeoutError("test did not release chat")
            return "old answer", True

        with patch("gui.QTimer.singleShot"), patch.object(VoiceWindow, "_create_tray"), patch("gui.sd.query_devices", return_value=[]), patch("gui.load_ui_preferences", return_value={"compact_pet_size": 138, "ui_animation_fps": 22, "wake": False, "barge_in": True}):
            window = VoiceWindow()
            window._stop_ui_timers()
            window.chat = SimpleNamespace(reply=reply)
            window.speaker = SimpleNamespace(begin_response=lambda: None, stop=lambda: None, enqueue=lambda text: None, wait_until_playing=lambda stopped: True)
            recognizer = SimpleNamespace(device=None, listen_once=listen_once)
            try:
                with patch.object(window, "_get_asr", return_value=recognizer), patch.object(window, "_queue_prompt") as submit:
                    window._start_chat("first question")
                    self.assertTrue(recognized.wait(2))
                    session = window.barge_session
                    self.assertTrue(session.done.wait(2))
                    session.thread.join(2)
                    self.app.processEvents()
                    self.assertFalse(session.consumed)
                    self.assertTrue(window.chat_running)
                    release_chat.set()
                    window.chat_thread.join(2)
                    self.assertFalse(window.chat_thread.is_alive())
                    self.app.processEvents()
                    submit.assert_called_once_with("next question")
                    self.assertTrue(session.consumed)
            finally:
                release_chat.set()
                window.chat_stop.set()
                window.barge_stop.set()
                if window.chat_thread:
                    window.chat_thread.join(2)
                window.exiting = True
                window._stop_ui_timers()
                window.voice_overlay.close()
                window.compact_window.close()
                window.close()
                window.deleteLater()

    def test_changing_fps_keeps_frames_and_animation_state(self):
        core = AssistantCore(138)
        try:
            core.set_pet_state("running")
            frames = core.pet_frames
            self.assertTrue(frames)
            core.pet_frame_index = 1
            with patch.object(core, "_load_pet_frames") as reload:
                core.set_animation_fps(60)
                core.set_animation_fps(10)
                reload.assert_not_called()
            self.assertIs(core.pet_frames, frames)
            self.assertEqual(core.pet_frame_index, 1)
            core.show()
            self.app.processEvents()
            self.assertTrue(core.timer.isActive())
            core.hide()
            self.assertFalse(core.timer.isActive())
        finally:
            core.close()

    def test_waveform_tracks_input_and_decays_to_silence(self):
        waveform = WaveformWidget()
        try:
            waveform.show()
            waveform.set_level(0.8)
            QTest.qWait(180)
            self.assertGreater(waveform.activity, 0.1)
            waveform.set_level(0.0)
            QTest.qWait(300)
            self.assertLess(waveform.activity, 0.05)
            waveform.hide()
            self.assertFalse(waveform.timer.isActive())
        finally:
            waveform.close()

    def test_each_ring_is_continuous_across_full_rotation(self):
        with patch("gui.core_style", return_value="A+D"):
            for widget in (AiCoreWidget(), AssistantCore(138)):
                try:
                    with patch("gui.QPainter") as factory:
                        painter = factory.return_value
                        widget.phase = 359.9
                        widget.paintEvent(None)
                        before = [call.args[1] / 16 for call in painter.drawArc.call_args_list]
                        painter.reset_mock()
                        widget.phase = 360.1
                        widget.paintEvent(None)
                        after = [call.args[1] / 16 for call in painter.drawArc.call_args_list]
                        self.assertGreater(len(before), 1)
                        self.assertEqual(len(before), len(after))
                        for start, end in zip(before, after):
                            self.assertLess(abs((end - start + 180) % 360 - 180), 1.0)
                finally:
                    widget.close()

    def test_settings_round_trip_restores_all_controls(self):
        windows = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            settings = Path(directory) / "ui-settings.json"
            stack.enter_context(patch("gui.ui_settings_path", return_value=settings))
            stack.enter_context(patch("gui.QTimer.singleShot"))
            stack.enter_context(patch.object(VoiceWindow, "_create_tray"))
            stack.enter_context(patch("gui.sd.query_devices", return_value=[{"name": "Test microphone", "max_input_channels": 1}]))
            try:
                first = VoiceWindow()
                windows.append(first)
                first._stop_ui_timers()
                for control in (first.tts_check, first.barge_check, first.wake_check, first.model_combo, first.device_combo):
                    control.blockSignals(True)
                first.compact_pet_size = 104
                first.ui_animation_fps = 30
                first.tts_check.setChecked(False)
                first.barge_check.setChecked(True)
                first.wake_check.setChecked(True)
                first.model_combo.setCurrentIndex(first.model_combo.count() - 1)
                first.device_combo.setCurrentIndex(1)
                first._save_ui_preferences()
                self.assertTrue(settings.is_file())
                second = VoiceWindow()
                windows.append(second)
                second._stop_ui_timers()
                self.assertEqual(second.pet_size_slider.value(), 104)
                self.assertEqual(second.animation_fps_slider.value(), 30)
                self.assertFalse(second.tts_check.isChecked())
                self.assertTrue(second.barge_check.isChecked())
                self.assertTrue(second.wake_check.isChecked())
                self.assertEqual(second.model_combo.currentData(), first.model_combo.currentData())
                self.assertEqual(second.device_combo.currentData(), 0)
                self.assertIsNone(second.asr)
            finally:
                for window in windows:
                    window.exiting = True
                    window._stop_ui_timers()
                    window.voice_overlay.close()
                    window.compact_window.close()
                    window.close()
                    window.deleteLater()

    def test_invalid_settings_use_defaults(self):
        for content in ('[]', 'null', '123', '{broken', '{"compact_pet_size":"bad"}'):
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content):
                preferences = load_ui_preferences()
                self.assertEqual(preferences["compact_pet_size"], 138)

    def test_microphone_change_does_not_reload_models(self):
        model = object()
        controller = SimpleNamespace(asr_lock=threading.Lock(), asr=model, asr_device=1)
        with patch("gui.StreamingASR") as factory:
            self.assertIs(VoiceWindow._get_asr(controller, 2), model)
            factory.assert_not_called()

    def test_pause_uses_explicit_operation_not_media_toggle(self):
        session = SimpleNamespace(try_pause_async=AsyncMock(return_value=True), try_play_async=AsyncMock(return_value=True))
        manager = SimpleNamespace(get_current_session=lambda: session)
        with patch("winsdk.windows.media.control.GlobalSystemMediaTransportControlsSessionManager.request_async", new=AsyncMock(return_value=manager)):
            result = LocalAgent()._media("pause")
            self.assertTrue(result.ok)
            session.try_pause_async.assert_awaited_once()
            session.try_play_async.assert_not_awaited()
