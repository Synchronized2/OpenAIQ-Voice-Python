from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

import config  # noqa: E402
from agent import LocalAction  # noqa: E402
from gui import CompactOrbWindow, VoiceWindow  # noqa: E402
from runtime_state import RuntimeState  # noqa: E402
from voices import EDGE_VOICES, normalize_voice  # noqa: E402


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.enterContext(patch("gui.load_ui_preferences", return_value={
            "compact_pet_size": 138, "ui_animation_fps": 22,
        }))

    def test_window_builds_with_models_and_controls(self) -> None:
        with patch.object(config, "WAKE_WORD_ENABLED", False), patch.object(
            config, "TTS_BARGE_IN_ENABLED", False
        ), patch.object(
            VoiceWindow, "_initialize_services", lambda window: window._set_status("就绪", "ready")
        ), patch.object(VoiceWindow, "_register_hotkey", lambda window: None):
            window = VoiceWindow()
            window.show()
            QTest.qWait(160)
            self.app.processEvents()
            try:
                self.assertEqual(window.windowTitle(), "OpenAIQ Voice")
                self.assertEqual(window.model_combo.count(), len(config.CHAT_MODEL_OPTIONS))
                self.assertEqual(window.model_combo.currentText(), config.CHAT_MODEL)
                self.assertEqual(window.tts_voice_combo.count(), len(EDGE_VOICES))
                self.assertEqual(
                    window.tts_voice_combo.currentData(), normalize_voice(config.TTS_VOICE)
                )
                for state, row in (
                    (RuntimeState.LISTENING, 8),
                    (RuntimeState.PROCESSING, 6),
                    (RuntimeState.EXECUTING, 7),
                    (RuntimeState.SPEAKING, 4),
                    (RuntimeState.ERROR, 5),
                ):
                    window._set_runtime_state(state)
                    self.assertIs(window.compact_window.core.pet_frames, window.compact_window.core.pet_rows[row])
                    self.assertIs(window.voice_overlay.core.pet_frames, window.voice_overlay.core.pet_rows[row])
                window._show_compact_layer()
                self.app.processEvents()
                self.assertTrue(window.compact_window.isVisible())
                self.assertFalse(window.isVisible())
                self.assertEqual(window.compact_window.status_label.text(), "准备中…")
                if config.UI_CORE_STYLE == "PET":
                    self.assertEqual(len(window.compact_window.core.pet_rows), 9)
                    self.assertGreaterEqual(len(window.compact_window.core.pet_frames), 4)
                window._set_status("出世中…", "model_loading")
                self.assertEqual(window.compact_window.status_label.text(), "出世中…")
                window.asr_model_loaded = True
                window._set_status("猴王出世", "model_ready")
                self.assertEqual(window.compact_window.status_label.text(), "猴王出世")
                with patch.object(window, "_show_voice_layer") as show_voice:
                    window._asr_finished("猴哥", True)
                    show_voice.assert_called_once_with()

                window._show_chat_layer()
                self.app.processEvents()
                self.assertTrue(window.send_button.isVisible())
                self.assertTrue(window.listen_compact.isVisible())
                self.assertTrue(window.live2d_view.isVisible())
                self.assertFalse(window.send_button.isEnabled())
                self.assertFalse(window.stop_button.isEnabled())
                window.input.setPlainText("测试")
                self.app.processEvents()
                self.assertTrue(window.send_button.isEnabled())
                window.input.clear()
                self.app.processEvents()
                self.assertFalse(window.send_button.isEnabled())
                self.assertGreater(window.device_combo.count(), 0)
                self.assertEqual(window.content_stack.currentIndex(), 0)
                self.assertEqual(
                    window.barge_check.isChecked(),
                    bool(config.TTS_BARGE_IN_ENABLED),
                )

                window._show_conversation()
                self.app.processEvents()
                self.assertEqual(window.content_stack.currentIndex(), 1)
                self.assertFalse(window.listen_compact.icon().isNull())

                with patch.object(window, "_start_agent") as start_agent:
                    window._queue_prompt("set volume to 33")
                    start_agent.assert_called_once_with(LocalAction("set_volume", 33))

                window.runtime.start("chat", RuntimeState.PROCESSING)
                window._show_voice_layer()
                self.app.processEvents()
                self.assertTrue(window.voice_overlay.isVisible())
                self.assertTrue(window.voice_overlay.waveform.isVisible())
                window._set_runtime_state(RuntimeState.SPEAKING)
                window._show_tts_segment("第一段正在朗读")
                window._show_tts_segment("第二段正在朗读")
                self.assertEqual(window.voice_overlay.answer_label.text(), "第二段正在朗读")
                for core in (
                    window.compact_window.core,
                    window.voice_overlay.core,
                ):
                    self.assertIs(core.pet_frames, core.pet_rows[4])
                window._show_barge_state("已打断", "请继续说完…")
                self.assertEqual(window.voice_overlay.answer_label.text(), "")
                window.runtime.finish("chat")

                window._asr_finished("拜拜", False)
                self.app.processEvents()
                self.assertEqual(window.current_mode, "compact")
                self.assertTrue(window.compact_window.isVisible())
                self.assertEqual(window.runtime.state, RuntimeState.STANDBY)

                window._show_chat_layer()
                window._toggle_settings()
                self.app.processEvents()
                self.assertTrue(window.settings_panel.isVisible())
                speaker = MagicMock()
                window.speaker = speaker
                window.tts_voice_combo.setCurrentIndex(2)
                window.tts_rate_slider.setValue(2)
                window.tts_volume_slider.setValue(65)
                speaker.configure.assert_called_with(
                    voice="zh-CN-YunxiNeural", rate="+12%", volume=0.65
                )
                window._preview_voice()
                window.preview_thread.join(timeout=2)
                self.app.processEvents()
                speaker.speak.assert_called_once()
                original_size = window.compact_pet_size
                original_fps = window.ui_animation_fps
                window.pet_size_slider.setValue(176)
                self.assertEqual(window.pet_size_value.text(), "176 px")
                self.assertEqual(
                    (window.compact_window.width(), window.compact_window.height()),
                    (192, 208),
                )
                window.animation_fps_slider.setValue(40)
                self.assertEqual(window.animation_fps_value.text(), "40 FPS")
                self.assertEqual(window.compact_window.core.timer.interval(), 25)
                self.assertEqual(window.voice_overlay.waveform.timer.interval(), 25)
                window.pet_size_slider.setValue(original_size)
                window.animation_fps_slider.setValue(original_fps)
                window.ui_settings_save_timer.stop()
                window._hide_settings()
                self.assertFalse(window.settings_panel.isVisible())
            finally:
                window.exiting = True
                window._stop_ui_timers()
                window.close()
                window.compact_window.close()
                window.voice_overlay.close()
                window.tray.hide()
                window.deleteLater()
                self.app.processEvents()

    def test_compact_size_animation_and_drag(self) -> None:
        calls = []
        controller = type(
            "CompactController",
            (),
            {"_show_chat_layer": lambda _self: calls.append("chat")},
        )()
        with patch.object(config, "COMPACT_PET_SIZE", 180), patch.object(
            config, "UI_ANIMATION_FPS", 40
        ):
            window = CompactOrbWindow(controller)
            self.assertEqual((window.width(), window.height()), (196, 212))
            self.assertEqual((window.core.width(), window.core.height()), (180, 180))
            self.assertEqual(window.core.timer.interval(), 25)
            window.show()
            self.app.processEvents()
            try:
                QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(110, 100))
                self.app.processEvents()
                self.assertEqual(calls, ["chat"])
                calls.clear()

                initial_position = window.pos()
                QTest.mousePress(window, Qt.LeftButton, pos=QPoint(110, 100))
                QTest.mouseMove(window, QPoint(150, 130))
                QTest.mouseRelease(window, Qt.LeftButton, pos=QPoint(150, 130))
                self.app.processEvents()
                self.assertEqual(calls, [])
                self.assertNotEqual(window.pos(), initial_position)
            finally:
                window.close()
                window.deleteLater()
                self.app.processEvents()

    def test_interrupted_chat_still_finishes_barge_in_monitor(self) -> None:
        class FakeChat:
            @staticmethod
            def reply(*_args, **_kwargs):
                return "旧回答", True

        class FakeSpeaker:
            @staticmethod
            def begin_response() -> None:
                pass

            @staticmethod
            def enqueue(_text: str) -> None:
                pass

            @staticmethod
            def stop() -> None:
                pass

        with patch.object(config, "WAKE_WORD_ENABLED", False), patch.object(
            config, "TTS_BARGE_IN_ENABLED", False
        ), patch.object(
            VoiceWindow, "_initialize_services", lambda _window: None
        ), patch.object(VoiceWindow, "_register_hotkey", lambda _window: None):
            window = VoiceWindow()
            window.chat = FakeChat()
            window.speaker = FakeSpeaker()
            window.barge_check.blockSignals(True)
            window.barge_check.setChecked(True)
            window.barge_check.blockSignals(False)
            finish_barge = MagicMock(return_value=(True, "下一个问题"))
            try:
                with patch.object(
                    window, "_start_barge_in_monitor", return_value=True
                ), patch.object(window, "_finish_barge_in_monitor", finish_barge):
                    window._start_chat("第一个问题")
                    window.chat_thread.join(timeout=2.0)
                self.assertFalse(window.chat_thread.is_alive())
                finish_barge.assert_called_once_with(True)
            finally:
                window.exiting = True
                window._stop_ui_timers()
                window.close()
                window.compact_window.close()
                window.voice_overlay.close()
                window.tray.hide()
                window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
