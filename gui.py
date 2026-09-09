from __future__ import annotations

import math
import ctypes
import ctypes.wintypes
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import sounddevice as sd
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizeGrip,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import config
from agent import LocalAction, LocalAgent, is_voice_exit_phrase, parse_local_intent
from asr import StreamingASR
from chat import ChatSession
from diagnostics import configure_logging, log
from runtime_state import RuntimeMachine, RuntimeState
from tts import EdgeSpeaker
from voice_session import BargeSession
from voices import (
    DEFAULT_EDGE_VOICE,
    EDGE_VOICES,
    clamp_rate_step,
    clamp_volume,
    edge_rate,
    normalize_voice,
)
from wake import WakeWordDetector
from live2d_models import Live2DModel, default_model_root, model_by_id, scan_live2d_models
from live2d_viewer import Live2DView


APP_USER_MODEL_ID = "OpenAIQ.Voice"
UI_SETTINGS_FILE = "ui-settings.json"


def clamp_int(value: object, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError, OverflowError):
        return fallback


def default_pet_size() -> int:
    return clamp_int(getattr(config, "COMPACT_PET_SIZE", 138), 72, 220, 138)


def default_animation_fps() -> int:
    return clamp_int(getattr(config, "UI_ANIMATION_FPS", 22), 10, 60, 22)


def default_tts_voice() -> str:
    return normalize_voice(getattr(config, "TTS_VOICE", DEFAULT_EDGE_VOICE))


def default_tts_rate_step() -> int:
    value = str(getattr(config, "TTS_RATE", "+0%")).strip().removesuffix("%")
    try:
        return clamp_rate_step(round(int(value) / 6))
    except ValueError:
        return 0


def default_tts_volume_percent() -> int:
    return round(clamp_volume(getattr(config, "TTS_VOLUME", 1.0)) * 100)


def default_live2d_root() -> Path:
    configured = getattr(config, "LIVE2D_MODEL_DIR", "")
    return Path(configured).expanduser() if configured else default_model_root()


def animation_interval_ms(fps: int) -> int:
    return max(16, min(100, round(1000 / clamp_int(fps, 10, 60, 22))))


def ui_settings_path() -> Path:
    base = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    return base / UI_SETTINGS_FILE


def load_ui_preferences() -> dict:
    try:
        data = json.loads(ui_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "compact_pet_size": clamp_int(
            data.get("compact_pet_size"), 72, 220, default_pet_size()
        ),
        "ui_animation_fps": clamp_int(
            data.get("ui_animation_fps"), 10, 60, default_animation_fps()
        ),
        "tts_voice": normalize_voice(data.get("tts_voice"), default_tts_voice()),
        "tts_rate": clamp_rate_step(data.get("tts_rate"), default_tts_rate_step()),
        "tts_volume": clamp_int(
            data.get("tts_volume"), 0, 100, default_tts_volume_percent()
        ),
        **{key: value for key, value in data.items()
           if key in {"speak", "barge_in", "wake"} and type(value) is bool},
        **{key: value for key, value in data.items()
           if key in {"model", "device_name"} and isinstance(value, str)},
        **{key: value for key, value in data.items()
           if key in {"live2d_root", "live2d_model"} and isinstance(value, str)},
    }


def core_style() -> str:
    value = str(getattr(config, "UI_CORE_STYLE", "A+D")).upper().replace(" ", "")
    return value if value in {"PET", "A", "D", "A+D"} else "A+D"


class UiSignals(QObject):
    status = Signal(str, str)
    runtime_state = Signal(object)
    asr_partial = Signal(str, object)
    asr_status = Signal(str, object)
    asr_finished = Signal(str, bool, object)
    barge_in_finished = Signal(str, object)
    barge_state = Signal(str, str, object)
    barge_monitor_finished = Signal(object)
    audio_level = Signal(float, object)
    tts_segment = Signal(str)
    chat_delta = Signal(str)
    chat_finished = Signal(str, bool, float)
    agent_finished = Signal(str, bool, float)
    error = Signal(str)
    worker_finished = Signal(str)


class ChatInput(QTextEdit):
    submit = Signal()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submit.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class OrbLogo(QWidget):
    def __init__(self, size: int = 28) -> None:
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        glow = QRadialGradient(rect.center(), rect.width() / 2)
        glow.setColorAt(0, QColor("#b8f4ff"))
        glow.setColorAt(0.35, QColor("#4d9dff"))
        glow.setColorAt(1, QColor("#4058e7"))
        painter.setBrush(glow)
        painter.setPen(QPen(QColor(170, 223, 255, 190), 1))
        painter.drawEllipse(rect)
        painter.setBrush(QColor(255, 255, 255, 205))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(rect.x() + 6, rect.y() + 4, 4, 4))


def build_line_icon(kind: str, color: str = "#8aaeff") -> QIcon:
    """Draw one icon from the app's shared 24 px optical grid."""
    pixmap = QPixmap(48, 48)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 1.82)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    if kind == "chat":
        painter.drawRoundedRect(QRectF(3.75, 4.25, 16.5, 13.25), 3.5, 3.5)
        path = QPainterPath(QPointF(8.2, 17.35))
        path.lineTo(6.55, 20.3)
        path.lineTo(11.05, 17.5)
        painter.drawPath(path)
    elif kind == "mic":
        painter.drawRoundedRect(QRectF(8.25, 2.75, 7.5, 12.5), 3.75, 3.75)
        path = QPainterPath(QPointF(5.1, 10.9))
        path.cubicTo(5.1, 16.2, 8.05, 19.0, 12, 19.0)
        path.cubicTo(15.95, 19.0, 18.9, 16.2, 18.9, 10.9)
        painter.drawPath(path)
        painter.drawLine(QPointF(12, 19), QPointF(12, 21.4))
        painter.drawLine(QPointF(8.7, 21.4), QPointF(15.3, 21.4))
    elif kind == "model":
        path = QPainterPath(QPointF(11.5, 3.5))
        path.cubicTo(11.15, 8.0, 8.95, 10.2, 4.5, 10.5)
        path.cubicTo(8.95, 10.8, 11.15, 13.0, 11.5, 17.5)
        path.cubicTo(11.85, 13.0, 14.05, 10.8, 18.5, 10.5)
        path.cubicTo(14.05, 10.2, 11.85, 8.0, 11.5, 3.5)
        painter.drawPath(path)
        small = QPainterPath(QPointF(18.5, 15.5))
        small.cubicTo(18.35, 17.45, 17.45, 18.35, 15.5, 18.5)
        small.cubicTo(17.45, 18.65, 18.35, 19.55, 18.5, 21.5)
        small.cubicTo(18.65, 19.55, 19.55, 18.65, 21.5, 18.5)
        small.cubicTo(19.55, 18.35, 18.65, 17.45, 18.5, 15.5)
        painter.drawPath(small)
    elif kind == "settings":
        for y, knob_x in ((6.0, 9.0), (12.0, 15.0), (18.0, 7.0)):
            painter.drawLine(QPointF(4, y), QPointF(knob_x - 2.1, y))
            painter.drawLine(QPointF(knob_x + 2.1, y), QPointF(20, y))
            painter.drawEllipse(QRectF(knob_x - 2.0, y - 2.0, 4.0, 4.0))
    elif kind == "plus":
        painter.drawLine(QPointF(12, 5.5), QPointF(12, 18.5))
        painter.drawLine(QPointF(5.5, 12), QPointF(18.5, 12))
    elif kind == "stop":
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(7.1, 7.1, 9.8, 9.8), 2.0, 2.0)
    elif kind == "send":
        painter.drawLine(QPointF(12, 18.3), QPointF(12, 6.1))
        path = QPainterPath(QPointF(7.25, 10.8))
        path.lineTo(12, 6.05)
        path.lineTo(16.75, 10.8)
        painter.drawPath(path)
    elif kind == "minimize":
        painter.drawLine(QPointF(6.2, 12), QPointF(17.8, 12))
    elif kind == "maximize":
        painter.drawRoundedRect(QRectF(5.75, 5.75, 12.5, 12.5), 1.5, 1.5)
    elif kind == "restore":
        painter.drawRoundedRect(QRectF(7.4, 5.5, 11.0, 11.0), 1.3, 1.3)
        path = QPainterPath(QPointF(16.5, 16.5))
        path.lineTo(16.5, 18.5)
        path.lineTo(5.5, 18.5)
        path.lineTo(5.5, 7.5)
        path.lineTo(7.5, 7.5)
        painter.drawPath(path)
    elif kind == "close":
        painter.drawLine(QPointF(7, 7), QPointF(17, 17))
        painter.drawLine(QPointF(17, 7), QPointF(7, 17))
    painter.end()
    return QIcon(pixmap)


class IconButton(QPushButton):
    """A state-aware icon button with consistent geometry and icon colors."""

    def __init__(
        self,
        kind: str,
        text: str = "",
        icon_size: int = 18,
        default_color: str = "#8c98aa",
        hover_color: str = "#f4f7fc",
        active_color: str = "#73d7ff",
        disabled_color: str = "#4b5361",
    ) -> None:
        super().__init__(text)
        self.icon_kind = kind
        self.icon_colors = (default_color, hover_color, active_color, disabled_color)
        self._last_icon_color = ""
        self.setIconSize(QSize(icon_size, icon_size))
        self.setCursor(Qt.PointingHandCursor)

    def set_kind(self, kind: str) -> None:
        if self.icon_kind == kind:
            return
        self.icon_kind = kind
        self._last_icon_color = ""
        self.update()

    def _sync_icon(self) -> None:
        default, hover, active, disabled = self.icon_colors
        if not self.isEnabled():
            color = disabled
        elif self.isDown():
            color = active
        elif self.isChecked() or bool(self.property("listening")):
            color = active
        elif self.underMouse():
            color = hover
        else:
            color = default
        if color != self._last_icon_color:
            self._last_icon_color = color
            self.setIcon(build_line_icon(self.icon_kind, color))

    def paintEvent(self, event) -> None:  # noqa: ANN001
        self._sync_icon()
        super().paintEvent(event)


class ToggleSwitch(QCheckBox):
    """Compact painted switch that keeps QCheckBox keyboard behavior and signals."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.hovered = False
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(30)

    def sizeHint(self) -> QSize:
        width = 44 + self.fontMetrics().horizontalAdvance(self.text())
        return QSize(width, 30)

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        checked = self.isChecked()
        enabled = self.isEnabled()
        track = QRectF(1, (self.height() - 18) / 2, 32, 18)
        if not enabled:
            track_color, border_color = QColor("#171b22"), QColor("#292f39")
        elif checked:
            track_color, border_color = QColor("#315fcf"), QColor("#527be0")
        elif self.hovered:
            track_color, border_color = QColor("#272e39"), QColor("#46515f")
        else:
            track_color, border_color = QColor("#202630"), QColor("#353e4b")
        painter.setBrush(track_color)
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(track, 9, 9)

        knob_x = track.right() - 15 if checked else track.left() + 2
        knob = QRectF(knob_x, track.top() + 2, 14, 14)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#eaf5ff") if checked else QColor("#9099a8"))
        painter.drawEllipse(knob)

        text_color = "#d3d8e2" if enabled else "#626a77"
        painter.setPen(QColor(text_color))
        painter.drawText(
            QRectF(44, 0, self.width() - 44, self.height()),
            Qt.AlignVCenter | Qt.AlignLeft,
            self.text(),
        )
        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#56718c"), 1))
            painter.drawRoundedRect(track.adjusted(-2, -2, 2, 2), 11, 11)


class TitleBar(QFrame):
    def __init__(self, window: "VoiceWindow") -> None:
        super().__init__()
        self.window = window
        self.drag_position: QPoint | None = None
        self.setObjectName("titleBar")
        self.setFixedHeight(54)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(21, 0, 14, 0)
        layout.setSpacing(9)
        layout.addWidget(OrbLogo(24))
        title = QLabel("AI 助手")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        layout.addStretch()

        model_label = QLabel("模型")
        model_label.setObjectName("topModelLabel")
        layout.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("topModelCombo")
        self.model_combo.setFixedWidth(158)
        for model, description in config.CHAT_MODEL_OPTIONS:
            self.model_combo.addItem(model, model)
            self.model_combo.setItemData(self.model_combo.count() - 1, description, Qt.ToolTipRole)
        default_index = self.model_combo.findData(config.CHAT_MODEL)
        self.model_combo.setCurrentIndex(max(0, default_index))
        layout.addWidget(self.model_combo)
        layout.addSpacing(12)

        self.new_chat_button = IconButton("plus", icon_size=18)
        self.new_chat_button.setObjectName("titleToolButton")
        self.new_chat_button.setToolTip("新建对话")
        self.new_chat_button.setFixedSize(32, 32)
        self.settings_button = IconButton("settings", icon_size=18)
        self.settings_button.setObjectName("titleToolButton")
        self.settings_button.setToolTip("设置")
        self.settings_button.setCheckable(True)
        self.settings_button.setFixedSize(32, 32)
        layout.addWidget(self.new_chat_button)
        layout.addWidget(self.settings_button)
        layout.addSpacing(5)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_label = QLabel("正在初始化")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_label)
        layout.addSpacing(15)

        for kind, tip, handler, name in (
            ("minimize", "最小化", window.showMinimized, "windowButton"),
            ("maximize", "最大化 / 还原", window._toggle_maximize, "windowButton"),
            ("close", "关闭到托盘", window.close, "closeButton"),
        ):
            button = IconButton(kind, icon_size=16)
            button.setObjectName(name)
            button.setToolTip(tip)
            button.setFixedSize(34, 30)
            button.clicked.connect(handler)
            layout.addWidget(button)
            if kind == "maximize":
                self.maximize_button = button

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self.drag_position is not None and event.buttons() & Qt.LeftButton:
            if self.window.isMaximized():
                self.window.showNormal()
                self.drag_position = QPoint(self.window.width() // 2, 25)
            self.window.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self.drag_position = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.window._toggle_maximize()


class AnimationClock:
    def showEvent(self, event) -> None:
        self._last_tick = time.monotonic()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)

    def elapsed(self) -> float:
        now = time.monotonic()
        previous = getattr(self, "_last_tick", now)
        self._last_tick = now
        return min(0.25, max(0.0, now - previous))


class WaveformWidget(AnimationClock, QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.phase = 0.0
        self.activity = 0.0
        self.target_level = 0.0
        self.level_at = 0.0
        self.setMinimumSize(360, 66)
        self.setMaximumHeight(74)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.setInterval(animation_interval_ms(default_animation_fps()))

    def set_animation_fps(self, fps: int) -> None:
        self.timer.setInterval(animation_interval_ms(fps))

    def set_active(self, active: bool) -> None:
        if not active:
            self.set_level(0.0)

    def set_level(self, level: float) -> None:
        self.target_level = max(0.0, min(1.0, level))
        self.level_at = time.monotonic()

    def _animate(self) -> None:
        dt = self.elapsed()
        self.phase += 3.11 * dt
        target = self.target_level if time.monotonic() - self.level_at < 0.4 else 0.0
        self.activity += (target - self.activity) * (1 - math.exp(-dt * 16))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center_y = self.height() / 2
        bars = max(48, min(92, self.width() // 7))
        gap = self.width() / bars
        for index in range(bars):
            normalized = (index - bars / 2) / (bars / 2)
            envelope = math.exp(-2.7 * abs(normalized))
            ripple = 0.5 + 0.5 * math.sin(index * 0.71 + self.phase)
            fine = 0.5 + 0.5 * math.sin(index * 1.91 - self.phase * 1.4)
            height = 2 + envelope * (9 + 24 * ripple * fine) * self.activity
            x = (index + 0.5) * gap
            tone = QColor("#4d71ff") if index % 3 else QColor("#28a9ff")
            tone.setAlpha(100 + int(145 * envelope))
            pen = QPen(tone, 1.55)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(x, center_y - height, x, center_y + height)


class AiCoreWidget(AnimationClock, QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.phase = 0.0
        self.active = False
        self.setFixedSize(168, 168)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.setInterval(animation_interval_ms(default_animation_fps()))

    def set_animation_fps(self, fps: int) -> None:
        self.timer.setInterval(animation_interval_ms(fps))

    def set_active(self, active: bool) -> None:
        self.active = active

    def _animate(self) -> None:
        # Keep phase continuous; only normalize the value used for drawing.
        self.phase += (2.4 if self.active else 0.75) * self.elapsed() / 0.045
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        style = core_style()
        if style == "PET":
            style = "A+D"
        phase = self.phase
        intensity = 220 if self.active else 140
        painter.setBrush(QColor(11, 16, 28, 230))
        painter.setPen(QPen(QColor(57, 81, 130, 120), 1))
        painter.drawEllipse(center, 57, 57)
        rings = []
        if "A" in style:
            rings.extend(((70, 86, phase, "#2979ff"), (64, 58, -phase * 0.8 + 120, "#23c8ff"), (50, 110, phase * 0.55 + 210, "#536dfe")))
        if "D" in style:
            rings.extend(((73, 42, -phase * 0.65 + 30, "#ffc857"), (59, 72, phase * 0.42 + 190, "#e26a3f")))
        for radius, span, offset, color in rings:
            pen = QPen(QColor(color), 2 if self.active else 1.4)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(
                QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                int(math.fmod(offset, 360.0) * 16),
                span * 16,
            )
        center_color = "#7ac8ff" if style == "A" else "#ffd166" if style == "D" else "#bde8ff"
        painter.setPen(QColor(center_color))
        font = painter.font()
        font.setPixelSize(25)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, "AI")


class AssistantCore(AnimationClock, QPushButton):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.phase = 0.0
        self.active = False
        self.hovered = False
        self.pet_rows: list[list[QImage]] = []
        self.pet_frames: list[QImage] = []
        self.pet_frame_index = 0
        self.pet_elapsed_ms = 0
        self._scaled_frames: dict[int, QPixmap] = {}
        self._scaled_size = 0
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("打开 OpenAIQ")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.setInterval(animation_interval_ms(default_animation_fps()))
        if core_style() == "PET":
            self._load_pet_frames()

    def set_animation_fps(self, fps: int) -> None:
        self.timer.setInterval(animation_interval_ms(fps))

    def _load_pet_frames(self) -> None:
        path = Path(getattr(config, "PET_SPRITESHEET", ""))
        if not path.is_file():
            return
        image = QImage(str(path))
        columns = 8
        rows = 9
        if image.isNull() or image.width() < columns or image.height() < rows:
            return
        cell_width = image.width() // columns
        cell_height = image.height() // rows
        count = max(1, min(columns, int(getattr(config, "PET_FRAME_COUNT", columns))))
        self.pet_rows = []
        for row in range(rows):
            frames = [
                image.copy(index * cell_width, row * cell_height, cell_width, cell_height)
                for index in range(count)
            ]
            self.pet_rows.append([frame for frame in frames if self._frame_has_pixels(frame)])
        self.set_pet_state("idle")

    @staticmethod
    def _frame_has_pixels(frame: QImage) -> bool:
        step = 8
        return any(
            frame.pixelColor(x, y).alpha() > 16
            for y in range(0, frame.height(), step)
            for x in range(0, frame.width(), step)
        )

    def set_pet_state(self, state: str) -> None:
        if not self.pet_rows:
            return
        idle_row = max(
            0,
            min(len(self.pet_rows) - 1, int(getattr(config, "PET_IDLE_ROW", 0))),
        )
        rows = {
            "idle": idle_row,
            "waving": 3,
            "speaking": 4,
            "failed": 5,
            "waiting": 6,
            "running": 7,
            "review": 8,
            "listening": 8,
        }
        row = max(0, min(len(self.pet_rows) - 1, rows.get(state, rows["idle"])))
        if not self.pet_rows[row]:
            row = idle_row
        if not self.pet_rows[row]:
            return
        if self.pet_frames is not self.pet_rows[row]:
            self.pet_frames = self.pet_rows[row]
            self.pet_frame_index = 0
            self.pet_elapsed_ms = 0

    def set_active(self, active: bool) -> None:
        self.active = active
        self.update()

    def _animate(self) -> None:
        # Keep phase continuous; only normalize the value used for drawing.
        dt = self.elapsed()
        self.phase += (2.7 if self.active else 0.65) * dt / 0.045
        if self.pet_frames:
            self.pet_elapsed_ms += dt * 1000
            interval = max(16, min(200, int(getattr(config, "PET_FRAME_INTERVAL_MS", 125))))
            while self.pet_elapsed_ms >= interval:
                self.pet_elapsed_ms -= interval
                self.pet_frame_index = (self.pet_frame_index + 1) % len(self.pet_frames)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self.hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self.hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        style = core_style()
        if style == "PET" and self.pet_frames:
            frame = self.pet_frames[self.pet_frame_index]
            if self._scaled_size != self.width():
                self._scaled_frames.clear()
                self._scaled_size = self.width()
            key = frame.cacheKey()
            if key not in self._scaled_frames:
                self._scaled_frames[key] = QPixmap.fromImage(frame).scaled(
                    int(self.width() * 0.96), int(self.height() * 0.96),
                    Qt.KeepAspectRatio, Qt.FastTransformation,
                )
            target = self._scaled_frames[key]
            painter.drawPixmap(
                int(center.x() - target.width() / 2),
                int(center.y() - target.height() / 2),
                target,
            )
            return
        if style == "PET":
            style = "A+D"
        radius = min(self.width(), self.height()) * 0.29
        phase = self.phase
        pulse = 1.5 + math.sin(math.radians(phase * 2)) * 1.2
        if self.active or self.hovered:
            halo = "#ffc857" if style == "D" else "#31c7ff"
            painter.setPen(QPen(QColor(halo), 5 + pulse))
            painter.drawEllipse(center, radius + 7, radius + 7)
        painter.setBrush(QColor("#111722"))
        border = "#ffc857" if style == "D" else "#2f73ff"
        painter.setPen(QPen(QColor(border), 2))
        painter.drawEllipse(center, radius, radius)
        rings = []
        if "A" in style:
            rings.extend(((0.12, 92, phase, "#2d7dff"), (0.03, 64, 135 - phase * 0.8, "#31d5ff")))
        if "D" in style:
            rings.extend(((0.16, 54, -phase * 0.6 + 80, "#ffc857"), (0.07, 38, phase * 0.45 + 220, "#e26a3f")))
        for inset, span, offset, color in rings:
            ring = radius + self.width() * inset
            pen = QPen(QColor(color), 1.8)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(
                QRectF(center.x() - ring, center.y() - ring, ring * 2, ring * 2),
                int(math.fmod(offset, 360.0) * 16),
                span * 16,
            )
        painter.setPen(QColor("#ffe6a3" if style == "D" else "#d8f4ff"))
        font = painter.font()
        font.setPixelSize(max(12, int(self.width() * 0.18)))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, "AI")


class CompactOrbWindow(QWidget):
    def __init__(self, controller: "VoiceWindow", pet_size: int | None = None) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.controller = controller
        self.drag_position: QPoint | None = None
        self.drag_start: QPoint | None = None
        self.drag_moved = False
        self.setAttribute(Qt.WA_TranslucentBackground)
        pet_size = clamp_int(pet_size, 72, 220, default_pet_size())
        self.setFixedSize(pet_size + 16, pet_size + 32)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(0)
        self.core = AssistantCore(pet_size)
        # 由悬浮窗统一处理鼠标事件，才能区分“单击打开聊天”和“拖动位置”。
        self.core.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.core.setToolTip("打开完整聊天")
        layout.addWidget(self.core)
        self.status_label = QLabel("准备中…")
        self.status_label.setObjectName("compactStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(22)
        self.status_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.status_label)

    def set_pet_size(self, pet_size: int) -> None:
        pet_size = clamp_int(pet_size, 72, 220, default_pet_size())
        self.core.setFixedSize(pet_size, pet_size)
        self.setFixedSize(pet_size + 16, pet_size + 32)
        self.updateGeometry()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            point = event.globalPosition().toPoint()
            self.drag_position = point - self.frameGeometry().topLeft()
            self.drag_start = point
            self.drag_moved = False
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self.drag_position is not None and event.buttons() & Qt.LeftButton:
            point = event.globalPosition().toPoint()
            if self.drag_start is not None and (point - self.drag_start).manhattanLength() >= 4:
                self.drag_moved = True
            self.move(point - self.drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        was_click = self.drag_position is not None and not self.drag_moved
        self.drag_position = None
        self.drag_start = None
        self.setCursor(Qt.OpenHandCursor)
        if event.button() == Qt.LeftButton and was_click:
            self.controller._show_chat_layer()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        state = {
            "出世中…": "running",
            "猴王出世": "waving",
            "出世失败": "failed",
        }.get(text, "idle")
        self.core.set_pet_state(state)

    def show_near_corner(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 28, screen.bottom() - self.height() - 34)
        self.show()
        self.raise_()


class VoiceOverlay(QWidget):
    def __init__(self, controller: "VoiceWindow") -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.controller = controller
        self.drag_position: QPoint | None = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(430, 360)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(9, 9, 9, 9)
        panel = QFrame()
        panel.setObjectName("voiceOverlayPanel")
        outer_layout.addWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 15, 22, 18)
        layout.setSpacing(5)

        header = QHBoxLayout()
        self.state_label = QLabel("我在听")
        self.state_label.setObjectName("overlayState")
        header.addWidget(self.state_label)
        header.addStretch()
        open_chat = IconButton("chat", icon_size=18, active_color="#a9e8ff")
        open_chat.setObjectName("overlayTool")
        open_chat.setToolTip("打开完整聊天")
        open_chat.setFixedSize(32, 32)
        open_chat.clicked.connect(controller._show_chat_layer)
        close = IconButton("close", icon_size=16)
        close.setObjectName("overlayTool")
        close.setToolTip("返回桌面")
        close.setFixedSize(32, 32)
        close.clicked.connect(controller._show_compact_layer)
        header.addWidget(open_chat)
        header.addWidget(close)
        layout.addLayout(header)

        self.core = AssistantCore(108)
        self.core.set_active(True)
        self.core.setToolTip("暂停或继续聆听")
        self.core.clicked.connect(controller._toggle_listening)
        layout.addWidget(self.core, 0, Qt.AlignCenter)
        self.waveform = WaveformWidget()
        self.waveform.setMinimumWidth(330)
        self.waveform.set_active(True)
        layout.addWidget(self.waveform, 0, Qt.AlignCenter)
        self.partial_label = QLabel("请说话…")
        self.partial_label.setObjectName("overlayPartial")
        self.partial_label.setAlignment(Qt.AlignCenter)
        self.partial_label.setWordWrap(True)
        self.partial_label.setMinimumHeight(30)
        layout.addWidget(self.partial_label)
        self.answer_label = QLabel("")
        self.answer_label.setObjectName("overlayAnswer")
        self.answer_label.setAlignment(Qt.AlignCenter)
        self.answer_label.setWordWrap(True)
        self.answer_label.setMinimumHeight(28)
        layout.addWidget(self.answer_label)

    def show_centered(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.center().x() - self.width() // 2
        y = screen.bottom() - self.height() - 54
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()

    def set_active(self, active: bool) -> None:
        self.core.set_active(active)
        self.waveform.set_active(active)
        self.state_label.setText("我在听" if active else "已暂停")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self.drag_position is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self.drag_position = None
        super().mouseReleaseEvent(event)


class ScenicBackground(QFrame):
    """Near-black ambient background inspired by dedicated voice consoles."""

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        background = QLinearGradient(0, 0, width, height)
        background.setColorAt(0, QColor("#0d1119"))
        background.setColorAt(0.58, QColor("#080b11"))
        background.setColorAt(1, QColor("#0b101c"))
        painter.fillRect(self.rect(), background)
        super().paintEvent(event)


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str = "") -> None:
        super().__init__()
        self.role = role
        self.setObjectName("userBubble" if role == "user" else "assistantBubble")
        self.set_compact(False)
        self.setMaximumWidth(720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 12, 17, 11)
        layout.setSpacing(6)
        role_label = QLabel("你" if role == "user" else "OPENAIQ")
        role_label.setObjectName("bubbleRole")
        self.text_label = QLabel(text)
        self.text_label.setObjectName("bubbleText")
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("bubbleMeta")
        self.meta_label.setWordWrap(True)
        self.meta_label.hide()
        layout.addWidget(role_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.meta_label)

    def append_text(self, text: str) -> None:
        self.text_label.setText(self.text_label.text() + text)

    def set_meta(self, text: str) -> None:
        self.meta_label.setText(text)
        self.meta_label.setVisible(bool(text))

    def set_compact(self, compact: bool) -> None:
        self.setMinimumWidth(220 if compact else (270 if self.role == "user" else 410))


def nav_button(text: str, icon: str, checked: bool = False) -> QPushButton:
    button = IconButton(icon, text=text, icon_size=19)
    button.setObjectName("navButton")
    button.setCheckable(True)
    button.setChecked(checked)
    button.setMinimumHeight(43)
    return button


class VoiceWindow(QMainWindow):
    @property
    def asr_running(self) -> bool:
        return self.runtime.is_running("asr")

    @property
    def chat_running(self) -> bool:
        return self.runtime.is_running("chat")

    @property
    def agent_running(self) -> bool:
        return self.runtime.is_running("agent")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenAIQ Voice")
        self.resize(900, 680)
        self.setMinimumSize(720, 560)
        self.setWindowIcon(build_icon())
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.signals = UiSignals()
        self.chat: ChatSession | None = None
        self.agent = LocalAgent()
        self.speaker: EdgeSpeaker | None = None
        self.asr: StreamingASR | None = None
        self.asr_device: int | None = None
        self.asr_model_loaded = False
        self._model_status_token = 0
        self.asr_thread: threading.Thread | None = None
        self.chat_thread: threading.Thread | None = None
        self.agent_thread: threading.Thread | None = None
        self.preview_thread: threading.Thread | None = None
        self.barge_thread: threading.Thread | None = None
        self.barge_session: BargeSession | None = None
        self.asr_preload_thread: threading.Thread | None = None
        self.asr_stop = threading.Event()
        self.barge_stop = threading.Event()
        self.chat_stop = threading.Event()
        self.asr_lock = threading.Lock()
        self.asr_session_lock = threading.Lock()
        self.voice_enabled = False
        self.runtime = RuntimeMachine()
        self.pending_prompts: deque[str] = deque()
        self.current_assistant: MessageBubble | None = None
        self.exiting = False
        self.has_error = False
        self.chat_started = False
        self.chat_wait_started = 0.0
        self.chat_wait_timer = QTimer(self)
        self.chat_wait_timer.setInterval(1000)
        self.chat_wait_timer.timeout.connect(self._update_chat_wait_status)
        self.current_mode = "chat"
        self.hotkey_registered = False
        self.wake_enabled = bool(getattr(config, "WAKE_WORD_ENABLED", False))
        self.wake_phrases = tuple(getattr(config, "WAKE_WORDS", ("孙悟空", "悟空")))
        self.wake_detector = WakeWordDetector(
            self.wake_phrases,
            tuple(getattr(config, "WAKE_WORD_ALIASES", ())),
            cooldown_seconds=float(getattr(config, "WAKE_COOLDOWN_SECONDS", 4.0)),
            min_chars=int(getattr(config, "WAKE_MIN_CHARS", 2)),
            max_chars=int(getattr(config, "WAKE_MAX_CHARS", 12)),
        )
        self.asr_wake_only = False
        self.ui_preferences = load_ui_preferences()
        self.wake_enabled = self.ui_preferences.get("wake", self.wake_enabled)
        self.compact_pet_size = self.ui_preferences["compact_pet_size"]
        self.ui_animation_fps = self.ui_preferences["ui_animation_fps"]
        configured_live2d_root = self.ui_preferences.get("live2d_root")
        self.live2d_root = Path(configured_live2d_root).expanduser() if configured_live2d_root else default_live2d_root()
        self.live2d_models = scan_live2d_models(self.live2d_root)
        preferred_live2d = self.ui_preferences.get("live2d_model")
        self.live2d_model = model_by_id(self.live2d_models, preferred_live2d)
        if self.live2d_model is None:
            self.live2d_model = next(
                (model for model in self.live2d_models if "hiyori" in model.name.lower()),
                self.live2d_models[0] if self.live2d_models else None,
            )
        self.ui_settings_save_timer = QTimer(self)
        self.ui_settings_save_timer.setSingleShot(True)
        self.ui_settings_save_timer.timeout.connect(self._save_ui_preferences)
        self.barge_preload_timer = QTimer(self)
        self.barge_preload_timer.setSingleShot(True)
        self.barge_preload_timer.timeout.connect(
            lambda: self._barge_setting_changed(True)
        )

        self._build_ui()
        self.compact_window = CompactOrbWindow(self, self.compact_pet_size)
        self.compact_window.set_status("准备中…")
        self.voice_overlay = VoiceOverlay(self)
        self._apply_animation_fps(self.ui_animation_fps)
        self.compact_window.setStyleSheet(APP_STYLE)
        self.voice_overlay.setStyleSheet(APP_STYLE)
        self._connect_signals()
        self._load_devices()
        self._create_tray()
        QTimer.singleShot(0, self._initialize_services)
        QTimer.singleShot(80, self._show_compact_layer)
        QTimer.singleShot(120, self._register_hotkey)
        if self.barge_check.isChecked():
            self.barge_preload_timer.start(300)

    def _build_ui(self) -> None:
        outer = QWidget()
        outer.setObjectName("outer")
        self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(9, 9, 9, 9)
        shell = QFrame()
        shell.setObjectName("shell")
        outer_layout.addWidget(shell)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.title_bar = TitleBar(self)
        self.status_dot = self.title_bar.status_dot
        self.status_label = self.title_bar.status_label
        self.model_combo = self.title_bar.model_combo
        preferred_model = self.model_combo.findData(self.ui_preferences.get("model", config.CHAT_MODEL))
        if preferred_model >= 0:
            self.model_combo.setCurrentIndex(preferred_model)
        self.clear_button = self.title_bar.new_chat_button
        self.settings_nav = self.title_bar.settings_button
        self.recent_label = QLabel("新的语音对话")
        self.recent_label.hide()
        shell_layout.addWidget(self.title_bar)

        self.scene = ScenicBackground()
        self.scene.setObjectName("scene")
        scene_layout = QHBoxLayout(self.scene)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(0)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(58, 22, 58, 24)
        center_layout.setSpacing(14)
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("contentStack")
        self.content_stack.addWidget(self._build_welcome_page())
        self.content_stack.addWidget(self._build_messages_page())
        center_layout.addWidget(self.content_stack, 1)
        center_layout.addWidget(self._build_composer())
        scene_layout.addWidget(center, 1)
        self.settings_panel = self._build_settings_panel()
        self.settings_panel.hide()
        scene_layout.addWidget(self.settings_panel)
        shell_layout.addWidget(self.scene, 1)
        self.setStyleSheet(APP_STYLE)

    def _build_welcome_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("welcomePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 28, 24, 8)
        layout.setSpacing(4)
        layout.addStretch(2)
        self.ai_core = AiCoreWidget()
        self.live2d_view = Live2DView()
        self.live2d_view.setMinimumSize(300, 280)
        self.live2d_view.set_model(self.live2d_model)
        layout.addWidget(self.live2d_view, 1)
        self.ai_core.hide()
        layout.addSpacing(18)
        greeting = QLabel("你好，我是你的 AI 助手")
        greeting.setObjectName("greeting")
        greeting.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("随时可以开始一段自然的对话")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(greeting)
        layout.addWidget(subtitle)
        layout.addStretch(3)
        return page

    def _build_messages_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.messages_widget = QWidget()
        self.messages_widget.setObjectName("messagesWidget")
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(20, 18, 20, 18)
        self.messages_layout.setSpacing(13)
        self.messages_layout.addStretch()
        self.scroll.setWidget(self.messages_widget)
        layout.addWidget(self.scroll)
        return page

    def _build_composer(self) -> QFrame:
        composer = QFrame()
        composer.setObjectName("composer")
        composer.setMinimumHeight(72)
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(17, 10, 10, 10)
        composer_layout.setSpacing(8)
        self.input = ChatInput()
        self.input.setObjectName("chatInput")
        self.input.setPlaceholderText("问点什么…")
        self.input.setFixedHeight(49)
        composer_layout.addWidget(self.input, 1)
        self.listen_compact = IconButton(
            "mic", icon_size=19, default_color="#9aa8bc", active_color="#68ddff"
        )
        self.listen_compact.setObjectName("toolButton")
        self.listen_compact.setToolTip("开始或停止聆听")
        self.listen_compact.setFixedSize(38, 38)
        composer_layout.addWidget(self.listen_compact)
        self.stop_button = IconButton("stop", icon_size=16)
        self.stop_button.setObjectName("toolButton")
        self.stop_button.setToolTip("停止当前任务")
        self.stop_button.setFixedSize(38, 38)
        self.stop_button.setEnabled(False)
        composer_layout.addWidget(self.stop_button)
        self.send_button = IconButton(
            "send",
            icon_size=18,
            default_color="#ffffff",
            hover_color="#ffffff",
            active_color="#ffffff",
            disabled_color="#687181",
        )
        self.send_button.setObjectName("sendButton")
        self.send_button.setToolTip("发送")
        self.send_button.setFixedSize(40, 40)
        self.send_button.setEnabled(False)
        composer_layout.addWidget(self.send_button)
        self.size_grip = QSizeGrip(composer)
        self.size_grip.setFixedSize(12, 12)
        composer_layout.addWidget(self.size_grip, 0, Qt.AlignBottom)
        return composer

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("settingsPanel")
        panel.setFixedWidth(304)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 22, 10, 20)
        layout.setSpacing(11)
        heading = QLabel("设置")
        heading.setObjectName("panelHeading")
        close_button = IconButton("close", icon_size=16)
        close_button.setObjectName("panelClose")
        close_button.setFixedSize(30, 30)
        close_button.setToolTip("关闭设置")
        close_button.clicked.connect(self._hide_settings)
        title_row = QHBoxLayout()
        title_row.addWidget(heading)
        title_row.addStretch()
        title_row.addWidget(close_button)
        layout.addLayout(title_row)

        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setObjectName("settingsContent")
        settings = QVBoxLayout(content)
        settings.setContentsMargins(0, 8, 8, 0)
        settings.setSpacing(11)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        settings.addWidget(section_label("麦克风"))
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("deviceCombo")
        settings.addWidget(self.device_combo)
        self.device_combo.setToolTip("切换后使用新麦克风，已加载的语音模型会保留")
        settings.addSpacing(10)

        settings.addWidget(section_label("2D 形象"))
        self.live2d_model_combo = QComboBox()
        self.live2d_model_combo.setObjectName("live2dModelCombo")
        settings.addWidget(self.live2d_model_combo)
        self.live2d_model_hint = QLabel()
        self.live2d_model_hint.setObjectName("settingHint")
        self.live2d_model_hint.setWordWrap(True)
        settings.addWidget(self.live2d_model_hint)
        live2d_actions = QHBoxLayout()
        self.live2d_scan_button = QPushButton("扫描模型目录")
        self.live2d_scan_button.setObjectName("live2dScanButton")
        self.live2d_choose_button = QPushButton("选择目录")
        self.live2d_choose_button.setObjectName("live2dChooseButton")
        live2d_actions.addWidget(self.live2d_scan_button)
        live2d_actions.addWidget(self.live2d_choose_button)
        settings.addLayout(live2d_actions)
        self._refresh_live2d_controls()

        settings.addWidget(section_label("朗读声音"))
        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.setObjectName("ttsVoiceCombo")
        for voice in EDGE_VOICES:
            self.tts_voice_combo.addItem(voice.label, voice.id)
        selected_voice = self.tts_voice_combo.findData(
            self.ui_preferences.get("tts_voice", default_tts_voice())
        )
        self.tts_voice_combo.setCurrentIndex(max(0, selected_voice))
        settings.addWidget(self.tts_voice_combo)
        self.tts_voice_description = QLabel()
        self.tts_voice_description.setObjectName("settingHint")
        self.tts_voice_description.setWordWrap(True)
        settings.addWidget(self.tts_voice_description)

        rate_row = QHBoxLayout()
        rate_label = QLabel("语速")
        rate_label.setObjectName("settingName")
        self.tts_rate_value = QLabel()
        self.tts_rate_value.setObjectName("settingValue")
        rate_row.addWidget(rate_label)
        rate_row.addStretch()
        rate_row.addWidget(self.tts_rate_value)
        settings.addLayout(rate_row)
        self.tts_rate_slider = QSlider(Qt.Horizontal)
        self.tts_rate_slider.setObjectName("settingSlider")
        self.tts_rate_slider.setRange(-5, 5)
        self.tts_rate_slider.setValue(
            self.ui_preferences.get("tts_rate", default_tts_rate_step())
        )
        settings.addWidget(self.tts_rate_slider)

        volume_row = QHBoxLayout()
        volume_label = QLabel("音量")
        volume_label.setObjectName("settingName")
        self.tts_volume_value = QLabel()
        self.tts_volume_value.setObjectName("settingValue")
        volume_row.addWidget(volume_label)
        volume_row.addStretch()
        volume_row.addWidget(self.tts_volume_value)
        settings.addLayout(volume_row)
        self.tts_volume_slider = QSlider(Qt.Horizontal)
        self.tts_volume_slider.setObjectName("settingSlider")
        self.tts_volume_slider.setRange(0, 100)
        self.tts_volume_slider.setSingleStep(5)
        self.tts_volume_slider.setValue(
            self.ui_preferences.get("tts_volume", default_tts_volume_percent())
        )
        settings.addWidget(self.tts_volume_slider)
        self.voice_preview_button = QPushButton("试听当前音色")
        self.voice_preview_button.setObjectName("voicePreviewButton")
        settings.addWidget(self.voice_preview_button)
        self._refresh_voice_settings()

        settings.addSpacing(10)
        settings.addWidget(section_label("桌面宠物"))

        size_row = QHBoxLayout()
        size_label = QLabel("大小")
        size_label.setObjectName("settingName")
        self.pet_size_value = QLabel(f"{self.compact_pet_size} px")
        self.pet_size_value.setObjectName("settingValue")
        size_row.addWidget(size_label)
        size_row.addStretch()
        size_row.addWidget(self.pet_size_value)
        settings.addLayout(size_row)
        self.pet_size_slider = QSlider(Qt.Horizontal)
        self.pet_size_slider.setObjectName("settingSlider")
        self.pet_size_slider.setRange(72, 220)
        self.pet_size_slider.setSingleStep(4)
        self.pet_size_slider.setPageStep(16)
        self.pet_size_slider.setValue(self.compact_pet_size)
        self.pet_size_slider.setToolTip("调整桌面常驻宠物的显示大小")
        settings.addWidget(self.pet_size_slider)

        fps_row = QHBoxLayout()
        fps_label = QLabel("动画刷新率")
        fps_label.setObjectName("settingName")
        self.animation_fps_value = QLabel(f"{self.ui_animation_fps} FPS")
        self.animation_fps_value.setObjectName("settingValue")
        fps_row.addWidget(fps_label)
        fps_row.addStretch()
        fps_row.addWidget(self.animation_fps_value)
        settings.addLayout(fps_row)
        self.animation_fps_slider = QSlider(Qt.Horizontal)
        self.animation_fps_slider.setObjectName("settingSlider")
        self.animation_fps_slider.setRange(10, 60)
        self.animation_fps_slider.setSingleStep(1)
        self.animation_fps_slider.setPageStep(5)
        self.animation_fps_slider.setValue(self.ui_animation_fps)
        self.animation_fps_slider.setToolTip("数值越高越流畅，也会增加少量 CPU 占用")
        settings.addWidget(self.animation_fps_slider)
        performance_hint = QLabel("推荐 22–30 FPS；高刷新率会增加少量 CPU 占用。")
        performance_hint.setObjectName("settingHint")
        performance_hint.setWordWrap(True)
        settings.addWidget(performance_hint)
        settings.addSpacing(7)
        self.tts_check = ToggleSwitch("朗读 AI 回复")
        self.tts_check.setChecked(self.ui_preferences.get("speak", True))
        settings.addWidget(self.tts_check)
        self.barge_check = ToggleSwitch("允许说话打断朗读")
        self.barge_check.setChecked(self.ui_preferences.get("barge_in", bool(config.TTS_BARGE_IN_ENABLED)))
        self.barge_check.setToolTip("外放可能被扬声器回声触发，建议佩戴耳机")
        settings.addWidget(self.barge_check)
        self.wake_check = ToggleSwitch("启用“悟空”语音唤醒")
        self.wake_check.setChecked(self.wake_enabled)
        settings.addWidget(self.wake_check)
        settings.addStretch()
        model_info = QLabel("唤醒词识别仅在本机运行。关闭后，麦克风只会在你主动开始语音时使用。")
        model_info.setObjectName("technicalInfo")
        model_info.setWordWrap(True)
        settings.addWidget(model_info)
        return panel

    def _connect_signals(self) -> None:
        self.listen_compact.clicked.connect(self._toggle_listening)
        self.settings_nav.clicked.connect(self._toggle_settings)
        self.stop_button.clicked.connect(self._stop_all)
        self.clear_button.clicked.connect(self._clear_chat)
        self.send_button.clicked.connect(self._submit_input)
        self.input.submit.connect(self._submit_input)
        self.input.textChanged.connect(self._sync_send_button)
        self.model_combo.currentIndexChanged.connect(self._change_model)
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        self.live2d_model_combo.currentIndexChanged.connect(self._live2d_model_changed)
        self.live2d_scan_button.clicked.connect(self._scan_live2d_directory)
        self.live2d_choose_button.clicked.connect(self._choose_live2d_directory)
        self.tts_voice_combo.currentIndexChanged.connect(self._voice_settings_changed)
        self.tts_rate_slider.valueChanged.connect(self._voice_settings_changed)
        self.tts_volume_slider.valueChanged.connect(self._voice_settings_changed)
        self.voice_preview_button.clicked.connect(self._preview_voice)
        self.wake_check.toggled.connect(self._wake_setting_changed)
        self.barge_check.toggled.connect(self._barge_setting_changed)
        for control in (self.tts_check, self.barge_check, self.wake_check):
            control.toggled.connect(lambda _value: self.ui_settings_save_timer.start(250))
        self.model_combo.currentIndexChanged.connect(lambda _value: self.ui_settings_save_timer.start(250))
        self.device_combo.currentIndexChanged.connect(lambda _value: self.ui_settings_save_timer.start(250))
        self.live2d_model_combo.currentIndexChanged.connect(lambda _value: self.ui_settings_save_timer.start(250))
        self.pet_size_slider.valueChanged.connect(self._pet_size_changed)
        self.animation_fps_slider.valueChanged.connect(self._animation_fps_changed)
        self.signals.status.connect(self._set_status)
        self.signals.runtime_state.connect(self._set_runtime_state)
        self.signals.asr_partial.connect(self._show_partial)
        self.signals.asr_status.connect(self._show_asr_status)
        self.signals.asr_finished.connect(self._asr_finished)
        self.signals.barge_in_finished.connect(self._barge_in_finished)
        self.signals.barge_state.connect(self._show_barge_state)
        self.signals.barge_monitor_finished.connect(self._barge_monitor_finished)
        self.signals.audio_level.connect(self._show_audio_level)
        self.signals.tts_segment.connect(self._show_tts_segment)
        self.signals.chat_delta.connect(self._chat_delta)
        self.signals.chat_finished.connect(self._chat_finished)
        self.signals.agent_finished.connect(self._agent_finished)
        self.signals.error.connect(self._show_error)
        self.signals.worker_finished.connect(self._worker_finished)

    def _sync_send_button(self) -> None:
        self.send_button.setEnabled(bool(self.input.toPlainText().strip()))

    def _refresh_live2d_controls(self) -> None:
        self.live2d_model_combo.blockSignals(True)
        try:
            self.live2d_model_combo.clear()
            for model in self.live2d_models:
                self.live2d_model_combo.addItem(model.name, model.id)
            index = self.live2d_model_combo.findData(
                self.live2d_model.id if self.live2d_model else None
            )
            if index >= 0:
                self.live2d_model_combo.setCurrentIndex(index)
        finally:
            self.live2d_model_combo.blockSignals(False)
        if self.live2d_model:
            self.live2d_model_hint.setText(
                f"{self.live2d_model.relative_manifest}\n目录：{self.live2d_root}"
            )
        elif self.live2d_models:
            self.live2d_model_hint.setText("请选择一个模型")
        else:
            self.live2d_model_hint.setText(
                "未找到可用模型。把 Live2D 模型目录放入指定位置后点击扫描。"
            )

    def _live2d_model_changed(self, _index: int) -> None:
        selected = model_by_id(self.live2d_models, self.live2d_model_combo.currentData())
        self.live2d_model = selected
        self.live2d_view.set_model(selected)
        self._refresh_live2d_controls()
        self.ui_settings_save_timer.start(250)

    def _scan_live2d_directory(self) -> None:
        self.live2d_models = scan_live2d_models(self.live2d_root)
        preferred = self.live2d_model.id if self.live2d_model else self.ui_preferences.get("live2d_model")
        self.live2d_model = model_by_id(self.live2d_models, preferred)
        if self.live2d_model is None and self.live2d_models:
            self.live2d_model = self.live2d_models[0]
        self.live2d_view.set_model(self.live2d_model)
        self._refresh_live2d_controls()
        self.ui_settings_save_timer.start(250)

    def _choose_live2d_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择 Live2D 模型目录", str(self.live2d_root)
        )
        if not selected:
            return
        self.live2d_root = Path(selected).expanduser().resolve()
        self._scan_live2d_directory()

    def _pet_size_changed(self, value: int) -> None:
        self.compact_pet_size = clamp_int(value, 72, 220, default_pet_size())
        self.pet_size_value.setText(f"{self.compact_pet_size} px")
        self.compact_window.set_pet_size(self.compact_pet_size)
        self.ui_settings_save_timer.start(250)

    def _animation_fps_changed(self, value: int) -> None:
        self.ui_animation_fps = clamp_int(value, 10, 60, default_animation_fps())
        self.animation_fps_value.setText(f"{self.ui_animation_fps} FPS")
        self._apply_animation_fps(self.ui_animation_fps)
        self.ui_settings_save_timer.start(250)

    def _apply_animation_fps(self, fps: int) -> None:
        for animated in (
            self.ai_core,
            self.compact_window.core,
            self.voice_overlay.core,
            self.voice_overlay.waveform,
        ):
            animated.set_animation_fps(fps)

    def _speech_options(self) -> tuple[str, str, float]:
        return (
            normalize_voice(self.tts_voice_combo.currentData()),
            edge_rate(self.tts_rate_slider.value()),
            self.tts_volume_slider.value() / 100,
        )

    def _refresh_voice_settings(self) -> None:
        voice_id = normalize_voice(self.tts_voice_combo.currentData())
        voice = next(item for item in EDGE_VOICES if item.id == voice_id)
        self.tts_voice_description.setText(voice.description)
        self.tts_rate_value.setText(edge_rate(self.tts_rate_slider.value()))
        self.tts_volume_value.setText(f"{self.tts_volume_slider.value()}%")

    def _voice_settings_changed(self, _value=None) -> None:  # noqa: ANN001
        self._refresh_voice_settings()
        if self.speaker:
            voice, rate, volume = self._speech_options()
            self.speaker.configure(voice=voice, rate=rate, volume=volume)
        self.ui_settings_save_timer.start(250)

    def _preview_voice(self) -> None:
        if not self.speaker or not self._begin_worker("tts-preview", RuntimeState.SPEAKING):
            return

        def work() -> None:
            try:
                self.speaker.speak("你好，我是悟空。这个声音听起来怎么样？")
            except Exception as exc:
                self.signals.error.emit(f"音色试听失败：{exc}")
            finally:
                self.signals.worker_finished.emit("tts-preview")

        self.preview_thread = threading.Thread(
            target=work, name="gui-tts-preview", daemon=True
        )
        self.preview_thread.start()

    def _save_ui_preferences(self) -> None:
        data = {
            "compact_pet_size": self.compact_pet_size,
            "ui_animation_fps": self.ui_animation_fps,
            "speak": self.tts_check.isChecked(),
            "barge_in": self.barge_check.isChecked(),
            "wake": self.wake_check.isChecked(),
            "model": self.model_combo.currentData(),
            "device_name": self.device_combo.currentText(),
            "tts_voice": normalize_voice(self.tts_voice_combo.currentData()),
            "tts_rate": self.tts_rate_slider.value(),
            "tts_volume": self.tts_volume_slider.value(),
            "live2d_root": str(self.live2d_root),
            "live2d_model": self.live2d_model.id if self.live2d_model else "",
        }
        path = ui_settings_path()
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        except OSError as exc:
            self.statusBar().showMessage(f"界面设置保存失败：{exc}", 4000)

    def _initialize_services(self) -> None:
        try:
            self.chat = ChatSession(model=self.model_combo.currentData())
            voice, rate, volume = self._speech_options()
            self.speaker = EdgeSpeaker(
                on_segment_start=self.signals.tts_segment.emit,
                voice=voice,
                rate=rate,
                volume=volume,
            )
            self._set_runtime_state(RuntimeState.READY)
        except Exception as exc:
            self._show_error(str(exc))

    def _load_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("系统默认麦克风", None)
        try:
            for index, device in enumerate(sd.query_devices()):
                if int(device.get("max_input_channels", 0)) > 0:
                    self.device_combo.addItem(f"{index} · {device['name']}", index)
            preferred = self.device_combo.findData(config.ASR_DEVICE)
            saved = self.device_combo.findText(self.ui_preferences.get("device_name", ""))
            if saved >= 0:
                preferred = saved
            self.device_combo.setCurrentIndex(preferred if preferred >= 0 else 0)
        except Exception as exc:
            self.device_combo.addItem(f"设备读取失败：{exc}", None)
        finally:
            self.device_combo.blockSignals(False)

    def _create_tray(self) -> None:
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("OpenAIQ Voice")
        menu = QMenu()
        show_action = QAction("显示窗口", self)
        show_action.triggered.connect(self._restore_window)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._restore_window()
            if reason == QSystemTrayIcon.DoubleClick
            else None
        )
        self.tray.show()

    def _submit_input(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._queue_prompt(text)

    def _queue_prompt(self, text: str) -> None:
        if self.current_mode == "voice" and is_voice_exit_phrase(text):
            self.voice_overlay.partial_label.setText(f"“{text}”")
            self.voice_overlay.answer_label.setText("回头见。")
            self._show_compact_layer()
            return
        self.has_error = False
        self._show_conversation()
        self.recent_label.setText(text[:38] + ("…" if len(text) > 38 else ""))
        self._add_message("user", text)
        if self.chat_running or self.asr_running or self.agent_running:
            self.pending_prompts.append(text)
            self.chat_stop.set()
            self.asr_stop.set()
            self.barge_stop.set()
            if self.speaker:
                self.speaker.stop()
            self._set_status("正在切换到新问题", "busy")
            return
        self._dispatch_prompt(text)

    def _dispatch_prompt(self, text: str) -> None:
        action = parse_local_intent(text)
        if action is not None:
            self._start_agent(action)
        else:
            self._start_chat(text)

    def _start_agent(self, action: LocalAction) -> None:
        if not self._begin_worker("agent", RuntimeState.EXECUTING):
            return
        self.has_error = False
        self.chat_stop = threading.Event()
        self.current_assistant = self._add_message("assistant", "")
        speak = self.tts_check.isChecked() and self.speaker is not None
        barge_enabled = speak and self.barge_check.isChecked()
        selected_device = self.device_combo.currentData()

        def work() -> None:
            started = time.perf_counter()
            result = self.agent.execute(action)
            elapsed = time.perf_counter() - started
            self.signals.agent_finished.emit(result.message, result.ok, elapsed)
            if speak and not self.chat_stop.is_set():
                barge_active = None
                try:
                    self.signals.runtime_state.emit(RuntimeState.SPEAKING)
                    self.speaker.begin_response()
                    barge_active = self._start_barge_in_monitor(barge_enabled, selected_device)
                    self.speaker.enqueue(result.message)
                    self.speaker.wait_interruptible(self.chat_stop)
                    barge_triggered, barge_text = self._finish_barge_in_monitor(barge_active)
                    self.signals.tts_segment.emit("")
                    if barge_triggered:
                        self.signals.barge_in_finished.emit(barge_text, barge_active)
                except Exception as exc:
                    self.speaker.stop()
                    self._finish_barge_in_monitor(barge_active)
                    self.signals.tts_segment.emit("")
                    self.signals.error.emit(f"操作结果朗读失败：{exc}")
            self.signals.worker_finished.emit("agent")

        self.agent_thread = threading.Thread(target=work, name="gui-agent", daemon=True)
        self.agent_thread.start()

    def _start_chat(self, text: str) -> None:
        if not self.chat or not self.speaker:
            self._show_error("服务尚未初始化，请检查配置。")
            return
        if not self._begin_worker("chat", RuntimeState.PROCESSING):
            return
        self.has_error = False
        self.chat_stop = threading.Event()
        self._show_conversation()
        self.current_assistant = self._add_message("assistant", "")
        self.chat_wait_started = time.monotonic()
        self.chat_wait_timer.start()
        self._update_chat_wait_status()
        speak = self.tts_check.isChecked()
        barge_enabled = speak and self.barge_check.isChecked()
        selected_device = self.device_combo.currentData()

        def work() -> None:
            started = time.perf_counter()
            barge_active = None
            try:
                if speak:
                    self.speaker.begin_response()
                    barge_active = self._start_barge_in_monitor(barge_enabled, selected_device)
                answer, interrupted = self.chat.reply(
                    text,
                    on_sentence=self.speaker.enqueue if speak else None,
                    on_text=self.signals.chat_delta.emit,
                    should_stop=self.chat_stop.is_set,
                )
                if speak and not interrupted:
                    self.signals.runtime_state.emit(RuntimeState.SPEAKING)
                    completed = self.speaker.wait_interruptible(self.chat_stop)
                    interrupted = interrupted or not completed
                barge_triggered, barge_text = self._finish_barge_in_monitor(barge_active)
                interrupted = interrupted or barge_triggered
                if speak:
                    self.signals.tts_segment.emit("")
                self.signals.chat_finished.emit(answer, interrupted, time.perf_counter() - started)
                if barge_triggered:
                    self.signals.barge_in_finished.emit(barge_text, barge_active)
            except Exception as exc:
                if self.speaker:
                    self.speaker.stop()
                self._finish_barge_in_monitor(barge_active)
                self.signals.tts_segment.emit("")
                self.signals.error.emit(str(exc))
            finally:
                self.signals.worker_finished.emit("chat")

        self.chat_thread = threading.Thread(target=work, name="gui-chat", daemon=True)
        self.chat_thread.start()

    def _toggle_listening(self) -> None:
        if self.voice_enabled:
            self.voice_enabled = False
            self.asr_stop.set()
            self._sync_listening_ui(False)
            self._set_runtime_state(RuntimeState.READY)
            return
        self.voice_enabled = True
        self.has_error = False
        self._sync_listening_ui(True)
        if not self.chat_running:
            self._start_asr()

    def _sync_listening_ui(self, active: bool) -> None:
        self.ai_core.set_active(active)
        self.listen_compact.setProperty("listening", active)
        self.voice_overlay.set_active(active)
        self.compact_window.core.set_active(
            self.current_mode == "compact" and self.wake_enabled
        )
        self.listen_compact.style().unpolish(self.listen_compact)
        self.listen_compact.style().polish(self.listen_compact)

    def _start_asr(self, wake_only: bool | None = None) -> None:
        if wake_only is None:
            wake_only = self.current_mode == "compact" and self.wake_enabled
        if self.barge_thread is not None and self.barge_thread.is_alive():
            return
        if self.runtime.busy:
            return
        if not wake_only and not self.voice_enabled:
            return
        if not self._begin_worker(
            "asr", RuntimeState.STANDBY if wake_only else RuntimeState.LISTENING
        ):
            return
        self.asr_wake_only = wake_only
        self.asr_stop = threading.Event()
        stopped = self.asr_stop
        selected_device = self.device_combo.currentData()
        needs_model_load = self.asr is None
        if needs_model_load:
            self._set_status("出世中…", "model_loading")
        if not wake_only:
            self.voice_overlay.partial_label.setText(
                "正在加载语音识别…" if needs_model_load else "请说话…"
            )

        def work() -> None:
            try:
                recognizer = self._get_asr(selected_device)
                if needs_model_load:
                    self.asr_model_loaded = True
                    self.signals.status.emit("猴王出世", "model_ready")
                with self.asr_session_lock:
                    recognizer.device = selected_device
                    text = recognizer.listen_once(
                        stop_event=stopped,
                        on_level=lambda value: self.signals.audio_level.emit(value, stopped),
                        on_partial=None if wake_only else lambda text: self.signals.asr_partial.emit(text, stopped),
                        on_status=None
                        if wake_only
                        else lambda value: self.signals.asr_status.emit(value, stopped),
                    )
                self.signals.asr_finished.emit(text, wake_only, stopped)
            except Exception as exc:
                if stopped.is_set():
                    return
                if needs_model_load:
                    self.asr_model_loaded = False
                    self.signals.status.emit("出世失败", "model_error")
                self.signals.error.emit(str(exc))
            finally:
                self.signals.worker_finished.emit("asr")

        self.asr_thread = threading.Thread(target=work, name="gui-asr", daemon=True)
        self.asr_thread.start()

    def _get_asr(self, selected_device: int | None) -> StreamingASR:
        with self.asr_lock:
            if self.asr is None:
                self.asr = StreamingASR(
                    model_dir=config.ASR_MODEL_DIR,
                    vad_model_dir=config.ASR_VAD_MODEL_DIR,
                    threads=config.ASR_THREADS,
                    device=selected_device,
                    vad_threshold=config.ASR_VAD_THRESHOLD,
                    vad_end_silence=config.ASR_VAD_END_SILENCE,
                    pre_roll=config.ASR_PRE_ROLL,
                    post_roll=config.ASR_POST_ROLL,
                    hotwords=config.ASR_HOTWORDS,
                )
                self.asr_device = selected_device
            return self.asr

    def _start_barge_in_monitor(self, enabled: bool, selected_device: int | None) -> BargeSession | None:
        if (
            not enabled
            or self.speaker is None
            or (self.barge_thread is not None and self.barge_thread.is_alive())
        ):
            return None
        session = BargeSession()
        log.info("barge.start id=%s", id(session))
        self.barge_session = session
        self.barge_stop = session.stopped
        chat_stopped = self.chat_stop

        def interrupt_playback() -> None:
            if session.stopped.is_set() or session.triggered.is_set():
                return
            session.triggered.set()
            log.info("barge.trigger id=%s", id(session))
            chat_stopped.set()
            if self.speaker:
                self.speaker.stop()
            self.signals.barge_state.emit("已打断", "请继续说完…", session)

        def recognition_status(value: str) -> None:
            if value == "正在识别":
                self.signals.barge_state.emit("正在识别", "正在识别新问题…", session)

        def work() -> None:
            try:
                # Warm the models before playback starts whenever LLM latency allows it.
                recognizer = self._get_asr(selected_device)
                if not self.speaker or not self.speaker.wait_until_playing(session.stopped):
                    return
                with self.asr_session_lock:
                    recognizer.device = selected_device
                    self.signals.barge_state.emit("可直接插话", "", session)
                    session.text = recognizer.listen_once(
                        stop_event=session.stopped,
                        on_level=lambda value: self.signals.audio_level.emit(value, session.stopped),
                        on_status=recognition_status,
                        on_speech_start=interrupt_playback,
                    )
            except Exception as exc:
                if not session.stopped.is_set():
                    self.signals.error.emit(f"语音打断监听失败：{exc}")
            finally:
                session.done.set()
                log.info("barge.end id=%s cancelled=%s chars=%s", id(session), session.stopped.is_set(), len(session.text))
                self.signals.barge_monitor_finished.emit(session)

        self.barge_thread = threading.Thread(target=work, name="gui-barge-in", daemon=True)
        session.thread = self.barge_thread
        self.barge_thread.start()
        return session

    def _finish_barge_in_monitor(self, session: BargeSession | None) -> tuple[bool, str]:
        if not session:
            return False, ""
        if not session.triggered.is_set():
            session.stopped.set()
        if session.thread:
            session.thread.join(timeout=25.0 if session.triggered.is_set() else 1.0)
        if not session.done.is_set():
            session.stopped.set()
        return session.take_result()

    def _show_partial(self, text: str, stopped=None) -> None:
        if stopped is not None and (stopped is not self.asr_stop or stopped.is_set()):
            return
        self.voice_overlay.partial_label.setText(text)

    def _show_asr_status(self, text: str, stopped: threading.Event) -> None:
        if stopped is not self.asr_stop or stopped.is_set():
            return
        self._set_status(text, "listening")
        if text == "正在聆听":
            self.voice_overlay.partial_label.setText("请说话…")

    def _show_audio_level(self, value: float, stopped: threading.Event) -> None:
        if stopped.is_set() or stopped not in (self.asr_stop, self.barge_stop):
            return
        self.voice_overlay.waveform.set_level(value)

    def _show_barge_state(self, state: str, hint: str, session=None) -> None:
        if session is not None and (session is not self.barge_session or session.stopped.is_set()):
            return
        if self.current_mode != "voice":
            return
        self.voice_overlay.state_label.setText(state)
        self.voice_overlay.partial_label.setText(hint)
        if state in {"已打断", "正在识别"}:
            self.voice_overlay.answer_label.clear()

    def _show_tts_segment(self, text: str) -> None:
        if text and not self.chat_stop.is_set() and (self.chat_running or self.agent_running):
            self.chat_wait_timer.stop()
            self._set_runtime_state(RuntimeState.SPEAKING)
        if self.current_mode == "voice":
            self.voice_overlay.answer_label.setText(text.strip())

    def _barge_monitor_finished(self, session: BargeSession) -> None:
        if session is not self.barge_session or self.exiting:
            return
        thread = session.thread
        if thread is not None and thread.is_alive():
            QTimer.singleShot(50, lambda: self._barge_monitor_finished(session))
            return
        if self.exiting or self.runtime.busy:
            return
        if self.voice_enabled and not self.asr_running:
            QTimer.singleShot(0, self._start_asr)
        elif self.current_mode == "compact" and self.wake_enabled and not self.asr_running:
            QTimer.singleShot(0, lambda: self._start_asr(True))

    def _asr_finished(self, text: str, wake_only: bool, stopped=None) -> None:
        if self.exiting or (stopped is not None and (stopped is not self.asr_stop or stopped.is_set())):
            return
        text = text.strip()
        if wake_only:
            if self.wake_detector.matches(text):
                self._show_voice_layer()
            return
        if text:
            self.voice_overlay.partial_label.setText(f"“{text}”")
            if is_voice_exit_phrase(text):
                self.voice_overlay.answer_label.setText("回头见。")
            else:
                self.voice_overlay.answer_label.setText("正在处理…")
            self._queue_prompt(text)
        elif self.voice_enabled and not self.asr_stop.is_set():
            self.voice_overlay.partial_label.setText("没有听清，请再说一次…")

    def _barge_in_finished(self, text: str, session=None) -> None:
        if self.exiting or (session is not None and (session is not self.barge_session or session.stopped.is_set())):
            return
        text = text.strip()
        if not text:
            if self.current_mode == "voice":
                self.voice_overlay.partial_label.setText("检测到说话，但没有听清…")
            return
        if self.current_mode == "voice":
            self.voice_overlay.partial_label.setText(f"“{text}”")
            self.voice_overlay.answer_label.setText("正在处理…")
        self._queue_prompt(text)

    def _update_chat_wait_status(self) -> None:
        if not self.chat_running:
            self.chat_wait_timer.stop()
            return
        if self.chat_stop.is_set():
            return
        elapsed = int(time.monotonic() - self.chat_wait_started)
        self._set_status(f"等待模型正文 · {elapsed} 秒", "busy")

    def _chat_delta(self, text: str) -> None:
        if self.chat_wait_timer.isActive():
            self.chat_wait_timer.stop()
            self._set_status("正在生成回复", "busy")
        if self.current_assistant:
            self.current_assistant.append_text(text)
            self._scroll_bottom()

    def _chat_finished(self, answer: str, interrupted: bool, elapsed: float) -> None:
        if self.current_assistant:
            if not self.current_assistant.text_label.text():
                self.current_assistant.text_label.setText("已中断" if interrupted else answer)
            timings = getattr(self.chat, "last_timings", {})
            first_token = timings.get("first_token")
            timing_text = f"全程 {elapsed:.2f} 秒（含朗读）"
            if first_token is not None:
                timing_text = f"首字 {first_token:.2f} 秒 · 模型 {timings['total']:.2f} 秒 · {timing_text}"
            self.current_assistant.set_meta(f"{'已中断 · ' if interrupted else ''}{timing_text}")
        self.current_assistant = None
        if self.current_mode == "voice" and not self.tts_check.isChecked():
            summary = answer.strip().replace("\n", " ")
            self.voice_overlay.answer_label.setText(summary[:90] + ("…" if len(summary) > 90 else ""))

    def _agent_finished(self, message: str, ok: bool, elapsed: float) -> None:
        if self.current_assistant:
            self.current_assistant.text_label.setText(message)
            self.current_assistant.set_meta(f"本地操作 · {elapsed:.2f} 秒")
        self.current_assistant = None
        if self.current_mode == "voice" and not self.tts_check.isChecked():
            self.voice_overlay.answer_label.setText(message)
        self._set_status("操作完成" if ok else "操作未执行", "ready" if ok else "error")

    def _worker_finished(self, kind: str) -> None:
        if kind == "chat":
            self.chat_wait_timer.stop()
        log.info("worker.end kind=%s", kind)
        if not self.runtime.finish(kind):
            return
        if kind == "asr":
            if self.has_error:
                self.voice_enabled = False
                self._sync_listening_ui(False)
        if self.exiting:
            return
        if self.pending_prompts and not self.chat_running and not self.asr_running and not self.agent_running:
            self._dispatch_prompt(self.pending_prompts.popleft())
        elif self.voice_enabled and not self.chat_running and not self.asr_running and not self.agent_running:
            self._set_runtime_state(RuntimeState.READY)
            QTimer.singleShot(180, self._start_asr)
        elif (
            self.current_mode == "compact"
            and self.wake_enabled
            and not self.chat_running
            and not self.asr_running
            and not self.agent_running
            and not self.has_error
        ):
            self._set_runtime_state(RuntimeState.STANDBY)
            QTimer.singleShot(250, lambda: self._start_asr(True))
        elif not self.chat_running and not self.asr_running and not self.agent_running and not self.has_error:
            self._set_runtime_state(
                RuntimeState.STANDBY if self.current_mode == "compact" else RuntimeState.READY
            )

    def _stop_all(self) -> None:
        self.chat_wait_timer.stop()
        was_busy = self.runtime.busy
        self.pending_prompts.clear()
        self.voice_enabled = False
        self._sync_listening_ui(False)
        self.asr_stop.set()
        self.barge_stop.set()
        self.chat_stop.set()
        if self.speaker:
            self.speaker.stop()
        self._set_runtime_state(
            RuntimeState.STOPPING
            if was_busy
            else RuntimeState.STANDBY
            if self.current_mode == "compact"
            else RuntimeState.READY
        )

    def _clear_chat(self) -> None:
        if self.chat_running or self.asr_running or self.agent_running:
            self._show_error("请先停止当前任务，再新建对话。")
            return
        if self.chat:
            self.chat.reset()
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.chat_started = False
        self.content_stack.setCurrentIndex(0)
        self.recent_label.setText("新的语音对话")
        self.voice_overlay.partial_label.setText("请说话…")
        self.voice_overlay.answer_label.clear()

    def _change_model(self) -> None:
        if self.chat:
            self.chat.model = self.model_combo.currentData()
            self._set_status(f"已切换至 {self.chat.model}", "ready")

    def _device_changed(self) -> None:
        self.asr_stop.set()
        self.barge_stop.set()
        if self.voice_enabled and not self.asr_running and not self.chat_running:
            self._start_asr()
        elif self.current_mode == "compact" and self.wake_enabled and not self.asr_running:
            self._start_asr(True)

    def _wake_setting_changed(self, enabled: bool) -> None:
        self.wake_enabled = enabled
        self.compact_window.core.set_active(enabled and self.current_mode == "compact")
        if not enabled and self.asr_running and self.asr_wake_only:
            self.asr_stop.set()
        elif enabled and self.current_mode == "compact" and not self.asr_running:
            self._start_asr(True)

    def _barge_setting_changed(self, enabled: bool) -> None:
        self.barge_preload_timer.stop()
        if self.exiting:
            return
        if not enabled:
            self.barge_stop.set()
            return
        if self.asr is not None:
            return
        if self.asr_preload_thread is not None and self.asr_preload_thread.is_alive():
            return
        selected_device = self.device_combo.currentData()

        def work() -> None:
            try:
                self._get_asr(selected_device)
                self.asr_model_loaded = True
                self.signals.status.emit("语音打断已就绪", "model_ready")
            except Exception as exc:
                self.signals.error.emit(f"语音打断准备失败：{exc}")

        self.asr_preload_thread = threading.Thread(
            target=work,
            name="gui-asr-preload",
            daemon=True,
        )
        self.asr_preload_thread.start()

    def _add_message(self, role: str, text: str) -> MessageBubble:
        bubble = MessageBubble(role, text)
        bubble.set_compact(self.settings_panel.isVisible())
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch()
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        QTimer.singleShot(0, self._scroll_bottom)
        return bubble

    def _scroll_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _show_conversation(self) -> None:
        self.chat_started = True
        self.content_stack.setCurrentIndex(1)

    def _show_chat(self) -> None:
        self.settings_panel.hide()
        self.settings_nav.setChecked(False)
        self.content_stack.setCurrentIndex(1 if self.chat_started else 0)

    def _show_compact_layer(self) -> None:
        if self.exiting:
            return
        self.current_mode = "compact"
        self.pending_prompts.clear()
        self.voice_enabled = False
        self.asr_stop.set()
        self.barge_stop.set()
        self.chat_stop.set()
        if self.speaker:
            self.speaker.stop()
        self._sync_listening_ui(False)
        self.hide()
        self.voice_overlay.hide()
        self.compact_window.show_near_corner()
        self._set_compact_model_status("待命" if self.asr_model_loaded else "准备中…")
        self._set_runtime_state(RuntimeState.STANDBY)
        if self.wake_enabled and not self.asr_running and not self.chat_running:
            if not self.asr_model_loaded:
                self._set_compact_model_status("出世中…")
            QTimer.singleShot(180, lambda: self._start_asr(True))

    def _show_voice_layer(self) -> None:
        if self.exiting:
            return
        wake_was_running = self.asr_running and self.asr_wake_only
        self.current_mode = "voice"
        self.has_error = False
        self.voice_enabled = True
        self.compact_window.hide()
        self.hide()
        self.voice_overlay.partial_label.setText("请说话…")
        self.voice_overlay.answer_label.clear()
        self.voice_overlay.show_centered()
        self._sync_listening_ui(True)
        self._set_runtime_state(RuntimeState.LISTENING)
        if wake_was_running:
            self.asr_stop.set()
        elif not self.asr_running and not self.chat_running:
            self._start_asr(False)

    def _show_chat_layer(self) -> None:
        if self.exiting:
            return
        from_voice = self.current_mode == "voice"
        self.current_mode = "chat"
        self.compact_window.hide()
        self.voice_overlay.hide()
        if not from_voice:
            self.voice_enabled = False
            if self.asr_running and self.asr_wake_only:
                self.asr_stop.set()
            self._sync_listening_ui(False)
        self._show_chat()
        self.showNormal()
        if not getattr(self, "chat_positioned", False):
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.center() - self.rect().center())
            self.chat_positioned = True
        self.show()
        self.raise_()
        self.activateWindow()
        if not self.runtime.busy:
            self._set_runtime_state(RuntimeState.READY)

    def _toggle_settings(self) -> None:
        visible = not self.settings_panel.isVisible()
        self.settings_panel.setVisible(visible)
        self.settings_nav.setChecked(visible)
        self._set_bubbles_compact(visible)

    def _hide_settings(self) -> None:
        self.settings_panel.hide()
        self.settings_nav.setChecked(False)
        self._set_bubbles_compact(False)

    def _set_bubbles_compact(self, compact: bool) -> None:
        for bubble in self.messages_widget.findChildren(MessageBubble):
            bubble.set_compact(compact)

    def _set_status(self, text: str, state: str = "ready") -> None:
        self.status_label.setText(text)
        if state == "model_loading":
            self._set_compact_model_status("出世中…")
        elif state == "model_ready":
            self._set_compact_model_status("猴王出世", transient=True)
        elif state == "model_error":
            self._set_compact_model_status("出世失败")
        if self.current_mode == "voice":
            overlay_states = {
                "busy": "正在思考",
                "listening": "我在听",
                "error": "出现问题",
                "ready": "已就绪",
            }
            self.voice_overlay.state_label.setText(overlay_states.get(state, text))
        colors = {"ready": "#62e6b2", "busy": "#ffd083", "listening": "#6fe7ff", "error": "#ff858f"}
        self.status_dot.setStyleSheet(f"color: {colors.get(state, '#94a5c2')};")

    def _set_compact_model_status(self, text: str, transient: bool = False) -> None:
        if not hasattr(self, "compact_window"):
            return
        self._model_status_token += 1
        token = self._model_status_token
        self.compact_window.set_status(text)
        if transient:
            QTimer.singleShot(
                3200,
                lambda: self.compact_window.set_status("待命")
                if token == self._model_status_token and self.asr_model_loaded
                else None,
            )

    def _begin_worker(self, worker: str, state: RuntimeState) -> bool:
        if not self.runtime.start(worker, state):
            return False
        self._set_runtime_state(state)
        log.info("worker.start kind=%s", worker)
        return True

    def _set_runtime_state(self, state: RuntimeState) -> None:
        self.runtime.transition(state)
        self.stop_button.setEnabled(self.runtime.busy and state != RuntimeState.STOPPING)
        labels = {
            RuntimeState.BOOTING: ("正在初始化", "busy", "正在启动"),
            RuntimeState.STANDBY: ("待命", "ready", "待命"),
            RuntimeState.READY: ("就绪", "ready", "已就绪"),
            RuntimeState.LISTENING: ("正在聆听", "listening", "我在听"),
            RuntimeState.PROCESSING: ("AI 正在思考", "busy", "正在思考"),
            RuntimeState.EXECUTING: ("正在执行本地操作", "busy", "正在执行"),
            RuntimeState.SPEAKING: ("正在朗读", "busy", "正在回答"),
            RuntimeState.STOPPING: ("正在停止", "busy", "正在停止"),
            RuntimeState.ERROR: ("发生错误", "error", "出现问题"),
        }
        text, tone, overlay = labels[state]
        self._set_status(text, tone)
        pet_state = {
            RuntimeState.BOOTING: "running",
            RuntimeState.STANDBY: "idle",
            RuntimeState.READY: "idle",
            RuntimeState.LISTENING: "listening",
            RuntimeState.PROCESSING: "waiting",
            RuntimeState.EXECUTING: "running",
            RuntimeState.SPEAKING: "speaking",
            RuntimeState.STOPPING: "waiting",
            RuntimeState.ERROR: "failed",
        }[state]
        for core in (self.compact_window.core, self.voice_overlay.core):
            core.set_pet_state(pet_state)
        live2d_state = {
            RuntimeState.BOOTING: "thinking",
            RuntimeState.STANDBY: "idle",
            RuntimeState.READY: "idle",
            RuntimeState.LISTENING: "listening",
            RuntimeState.PROCESSING: "thinking",
            RuntimeState.EXECUTING: "executing",
            RuntimeState.SPEAKING: "speaking",
            RuntimeState.STOPPING: "thinking",
            RuntimeState.ERROR: "error",
        }[state]
        if hasattr(self, "live2d_view"):
            self.live2d_view.set_state(live2d_state)
        voice_controls_enabled = not self.runtime.busy
        for control in (
            self.tts_voice_combo,
            self.tts_rate_slider,
            self.tts_volume_slider,
            self.voice_preview_button,
        ):
            control.setEnabled(voice_controls_enabled)
        if self.current_mode == "voice":
            self.voice_overlay.state_label.setText(overlay)

    def _show_error(self, message: str) -> None:
        self.has_error = True
        self._set_runtime_state(RuntimeState.ERROR)
        if self.current_mode == "compact" and self.asr_wake_only:
            self.wake_enabled = False
            self.wake_check.blockSignals(True)
            self.wake_check.setChecked(False)
            self.wake_check.blockSignals(False)
            self.compact_window.core.set_active(False)
            self.tray.showMessage(
                "OpenAIQ Voice",
                f"唤醒词监听已关闭：{message}",
                QSystemTrayIcon.Warning,
                4000,
            )
            return
        self._show_conversation()
        self._add_message("assistant", f"错误：{message}")

    def _toggle_maximize(self) -> None:
        self.showNormal() if self.isMaximized() else self.showMaximized()
        self.title_bar.maximize_button.set_kind(
            "restore" if self.isMaximized() else "maximize"
        )

    def _restore_window(self) -> None:
        self._show_chat_layer()

    def _register_hotkey(self) -> None:
        if sys.platform != "win32" or self.hotkey_registered:
            return
        # MOD_CONTROL | MOD_NOREPEAT, VK_SPACE
        self.hotkey_registered = bool(
            ctypes.windll.user32.RegisterHotKey(int(self.winId()), 1, 0x0002 | 0x4000, 0x20)
        )
        if not self.hotkey_registered:
            self.tray.showMessage(
                "OpenAIQ Voice",
                "Ctrl + Space 已被其他程序占用，仍可点击桌面 AI 核心使用。",
                QSystemTrayIcon.Warning,
                3500,
            )

    def nativeEvent(self, event_type, message):  # noqa: ANN001, N802
        if sys.platform == "win32":
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam == 1:
                QTimer.singleShot(0, self._show_voice_layer)
                return True, 0
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self.exiting:
            event.accept()
            return
        event.ignore()
        self._show_compact_layer()

    def _quit(self) -> None:
        self.exiting = True
        if self.ui_settings_save_timer.isActive():
            self.ui_settings_save_timer.stop()
            self._save_ui_preferences()
        self._stop_all()
        self.barge_stop.set()
        self._stop_ui_timers()
        if self.hotkey_registered and sys.platform == "win32":
            ctypes.windll.user32.UnregisterHotKey(int(self.winId()), 1)
            self.hotkey_registered = False
        if self.speaker:
            self.speaker.close()
        self.voice_overlay.hide()
        self.compact_window.hide()
        self.live2d_view.close()
        self.tray.hide()
        QApplication.quit()

    def _stop_ui_timers(self) -> None:
        self.chat_wait_timer.stop()
        self.barge_preload_timer.stop()
        self.ui_settings_save_timer.stop()
        for animated in (
            self.ai_core,
            self.compact_window.core,
            self.voice_overlay.core,
            self.voice_overlay.waveform,
        ):
            animated.timer.stop()


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def build_icon_pixmap(size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 128.0, size / 128.0)

    shell = QRadialGradient(QPointF(42, 30), 118)
    shell.setColorAt(0, QColor("#263a62"))
    shell.setColorAt(0.45, QColor("#111a2c"))
    shell.setColorAt(1, QColor("#070b13"))
    painter.setBrush(shell)
    painter.setPen(QPen(QColor(75, 111, 166, 205), 2.5))
    painter.drawRoundedRect(QRectF(5, 5, 118, 118), 27, 27)

    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(31, 124, 255, 40), 14))
    painter.drawEllipse(QRectF(28, 28, 72, 72))
    for start, span, color in (
        (15, 76, "#2478ff"),
        (132, 55, "#2ee3ff"),
        (220, 98, "#4966ff"),
    ):
        arc_pen = QPen(QColor(color), 6.5)
        arc_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(QRectF(26, 26, 76, 76), start * 16, span * 16)

    core = QRadialGradient(QPointF(57, 53), 31)
    core.setColorAt(0, QColor("#88f2ff"))
    core.setColorAt(0.34, QColor("#278dff"))
    core.setColorAt(1, QColor("#172b70"))
    painter.setBrush(core)
    painter.setPen(QPen(QColor(124, 225, 255, 220), 2.5))
    painter.drawEllipse(QRectF(41, 41, 46, 46))

    mark = QPainterPath(QPointF(64, 50))
    mark.lineTo(77, 64)
    mark.lineTo(64, 78)
    mark.lineTo(51, 64)
    mark.closeSubpath()
    painter.setBrush(QColor(7, 20, 45, 215))
    painter.setPen(QPen(QColor(211, 250, 255, 235), 2.4))
    painter.drawPath(mark)
    painter.setBrush(QColor("#eafbff"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QRectF(60.5, 60.5, 7, 7))
    painter.end()
    return pixmap


def build_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256):
        icon.addPixmap(build_icon_pixmap(size))
    return icon


def configure_windows_app_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


APP_STYLE = """
QWidget { color: #edf0f7; font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QWidget#outer { background: transparent; }
QFrame#shell { background: #090b10; border: 1px solid #282d37; border-radius: 14px; }
QFrame#titleBar { background: #0c0f15; border: 0; border-bottom: 1px solid #20242d; border-top-left-radius: 14px; border-top-right-radius: 14px; }
QLabel#appTitle { color: #f5f7fb; font-size: 16px; font-weight: 700; }
QLabel#edition { color: #789fff; font-size: 9px; font-weight: 700; background: #18233f; border-radius: 4px; padding: 3px 6px; }
QLabel#topModelLabel { color: #676f7e; font-size: 10px; }
QComboBox#topModelCombo { color: #cbd1dd; background: transparent; border: 0; border-radius: 5px; padding: 4px 8px; min-height: 24px; }
QComboBox#topModelCombo:hover { color: white; background: #171b23; }
QLabel#statusLabel { color: #7f8795; font-size: 10px; }
QLabel#statusDot { font-size: 12px; }
QPushButton#windowButton, QPushButton#closeButton, QPushButton#panelClose, QPushButton#titleToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 0; }
QPushButton#windowButton:hover, QPushButton#panelClose:hover, QPushButton#titleToolButton:hover { background: #181d26; border-color: #242b36; }
QPushButton#windowButton:pressed, QPushButton#panelClose:pressed, QPushButton#titleToolButton:pressed { background: #222936; border-color: #303a49; }
QPushButton#titleToolButton:checked { background: #172532; border-color: #244459; }
QPushButton#windowButton:focus, QPushButton#panelClose:focus, QPushButton#titleToolButton:focus { border-color: #47627f; }
QPushButton#closeButton:hover { background: #b93a4a; border-color: #d85866; }
QPushButton#closeButton:pressed { background: #952c39; border-color: #bd4653; }
QFrame#sidebar { background: #0c0f14; border: 0; border-right: 1px solid #20242c; border-bottom-left-radius: 14px; }
QPushButton#navButton { color: #8e96a5; background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 0 13px; text-align: left; font-size: 13px; }
QPushButton#navButton:hover { color: #f5f7fb; background: #151a22; border-color: #202630; }
QPushButton#navButton:pressed { background: #1c222c; }
QPushButton#navButton:checked { color: #ffffff; background: #18212b; border-color: #263644; }
QPushButton#navButton:focus { border-color: #3e5d78; }
QLabel#sectionLabel { color: #5e6675; font-size: 10px; font-weight: 600; padding: 2px 7px; }
QLabel#recentLabel { color: #8f97a6; font-size: 11px; background: transparent; border-radius: 6px; padding: 9px 11px; }
QLabel#recentLabel:hover { color: #d4d8e1; background: #12151c; }
QLabel#sidebarFooter { color: #464d59; font-size: 8px; padding: 4px; }
QPushButton#newChatButton { color: #d6dae4; background: #171a21; border: 1px solid #252a33; border-radius: 7px; text-align: left; padding: 0 14px; }
QPushButton#newChatButton:hover { color: white; background: #1d212a; border-color: #333946; }
QComboBox { color: #e5e8ef; background: #11151d; border: 1px solid #2a303b; border-radius: 7px; padding: 7px 9px; min-height: 27px; }
QComboBox:hover, QTextEdit:focus { border-color: #426bd1; }
QComboBox QAbstractItemView { color: #edf0f7; background: #12161e; border: 1px solid #303744; selection-background-color: #253963; outline: 0; }
QLabel#greeting { color: #f6f7fb; font-size: 24px; font-weight: 600; }
QLabel#welcomeSubtitle { color: #858d9c; font-size: 12px; }
QWidget#voiceDock { background: transparent; }
QLabel#partialLabel { color: #afb6c4; font-size: 13px; font-weight: 500; }
QPushButton#listenButton { background: #3f5ee8; border: 1px solid #657cf2; border-radius: 31px; padding: 0; }
QPushButton#listenButton:hover { background: #4d6bef; border-color: #8394f8; }
QPushButton#listenButton:pressed { background: #324bc7; border-color: #6174db; }
QPushButton#listenButton:focus { border-color: #b0bbff; }
QPushButton#listenButton[listening="true"] { background: #167cab; border-color: #5acff0; }
QLabel#listenHint { color: #626a78; font-size: 10px; margin-top: 2px; }
QFrame#composer { background: #10131a; border: 1px solid #282e39; border-radius: 8px; }
QTextEdit#chatInput { color: #eef0f5; background: transparent; border: 0; padding: 9px 4px; selection-background-color: #315ab8; font-size: 13px; }
QPushButton#toolButton { background: transparent; border: 1px solid transparent; border-radius: 19px; padding: 0; }
QPushButton#toolButton:hover { background: #1a202a; border-color: #252d38; }
QPushButton#toolButton:pressed { background: #222a36; border-color: #34404f; }
QPushButton#toolButton:focus { border-color: #43637d; }
QPushButton#toolButton:disabled { background: transparent; border-color: transparent; }
QPushButton#toolButton[listening="true"] { background: #122c3b; border-color: #1e5268; }
QPushButton#sendButton { background: #3d5fe8; border: 1px solid #627bef; border-radius: 20px; padding: 0; }
QPushButton#sendButton:hover { background: #4b6cf0; border-color: #8193f7; }
QPushButton#sendButton:pressed { background: #304bc5; border-color: #5970d8; }
QPushButton#sendButton:focus { border-color: #b0bdff; }
QPushButton#sendButton:disabled { background: #191d25; border-color: #282e38; }
QScrollArea { background: transparent; border: 0; }
QWidget#messagesWidget { background: transparent; }
QFrame#assistantBubble, QFrame#userBubble { background: transparent; border: 0; }
QLabel#bubbleRole { color: #6d85c7; font-size: 10px; font-weight: 600; }
QLabel#bubbleText { color: #e9ebf0; font-size: 14px; }
QLabel#bubbleMeta { color: #606977; font-size: 9px; }
QFrame#settingsPanel { background: #0d1118; border-left: 1px solid #252b35; border-bottom-right-radius: 14px; }
QWidget#settingsContent { background: transparent; }
QLabel#panelHeading { color: #f3f4f7; font-size: 16px; font-weight: 600; }
QLabel#settingHint, QLabel#technicalInfo { color: #687181; font-size: 10px; }
QLabel#settingName { color: #c3c9d4; font-size: 12px; }
QLabel#settingValue { color: #8fb8ff; font-size: 11px; font-weight: 600; }
QSlider#settingSlider { min-height: 20px; }
QSlider#settingSlider::groove:horizontal { height: 4px; background: #282f3a; border-radius: 2px; }
QSlider#settingSlider::sub-page:horizontal { background: #4f78e8; border-radius: 2px; }
QSlider#settingSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #edf4ff; border: 2px solid #5e87f3; border-radius: 7px; }
QSlider#settingSlider::handle:horizontal:hover { background: #ffffff; border-color: #7da2ff; }
QPushButton#voicePreviewButton { color: #dbe7ff; background: #18243a; border: 1px solid #31507d; border-radius: 7px; padding: 8px 10px; }
QPushButton#voicePreviewButton:hover { color: white; background: #203252; border-color: #4d73ad; }
QPushButton#voicePreviewButton:pressed { background: #142039; }
QPushButton#voicePreviewButton:disabled { color: #697181; background: #151920; border-color: #252b35; }
QCheckBox { color: #c0c5cf; spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QScrollBar:vertical { width: 7px; background: transparent; }
QScrollBar::handle:vertical { background: #343a46; min-height: 32px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: white; background: #181d26; border: 1px solid #383f4c; padding: 4px; }
QLabel#compactStatus { color: #8ea9c7; font-size: 9px; font-weight: 600; background: transparent; }
QFrame#voiceOverlayPanel { background: rgba(13, 16, 23, 245); border: 1px solid #303744; border-radius: 8px; }
QLabel#overlayState { color: #f1f3f8; font-size: 14px; font-weight: 600; }
QLabel#overlayPartial { color: #e3e6ee; font-size: 15px; }
QLabel#overlayAnswer { color: #7f8ca3; font-size: 11px; }
QPushButton#overlayTool { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 0; }
QPushButton#overlayTool:hover { background: #1b212b; border-color: #29323e; }
QPushButton#overlayTool:pressed { background: #252d39; border-color: #374453; }
QPushButton#overlayTool:focus { border-color: #48637d; }
"""


def main() -> int:
    configure_logging()
    configure_windows_app_identity()
    app = QApplication(sys.argv)
    app.setApplicationName("OpenAIQ Voice")
    app.setApplicationDisplayName("OpenAIQ Voice")
    app.setOrganizationName("OpenAIQ")
    app.setDesktopFileName("openaiq-voice")
    app.setQuitOnLastWindowClosed(False)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setWindowIcon(build_icon())

    def handle_exception(exc_type, exc_value, exc_traceback) -> None:  # noqa: ANN001
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        Path("openaiq-ui-error.log").write_text(details, encoding="utf-8")
        QMessageBox.critical(None, "OpenAIQ Voice", str(exc_value))

    sys.excepthook = handle_exception
    window = VoiceWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
