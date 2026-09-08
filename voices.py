from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeVoice:
    id: str
    label: str
    description: str


EDGE_VOICES = (
    EdgeVoice("zh-CN-XiaoxiaoNeural", "晓晓 · 女声", "自然亲切，适合日常陪伴与聊天（默认）"),
    EdgeVoice("zh-CN-XiaoyiNeural", "晓伊 · 女声", "清亮柔和，适合轻松交流"),
    EdgeVoice("zh-CN-YunxiNeural", "云希 · 男声", "年轻明朗，适合聊天与讲故事"),
    EdgeVoice("zh-CN-YunjianNeural", "云健 · 男声", "浑厚有力，适合叙述与有力度的表达"),
    EdgeVoice("zh-CN-YunyangNeural", "云扬 · 男声", "清晰稳重，适合讲解与长文朗读"),
    EdgeVoice("zh-TW-HsiaoChenNeural", "晓臻 · 女声／台湾普通话", "柔和舒缓，带台湾口音"),
)
DEFAULT_EDGE_VOICE = EDGE_VOICES[0].id
EDGE_VOICE_IDS = frozenset(voice.id for voice in EDGE_VOICES)


def normalize_voice(value: object, fallback: str = DEFAULT_EDGE_VOICE) -> str:
    voice = str(value or "").strip()
    return voice if voice in EDGE_VOICE_IDS else fallback


def clamp_rate_step(value: object, fallback: int = 0) -> int:
    try:
        return max(-5, min(5, int(value)))
    except (TypeError, ValueError, OverflowError):
        return fallback


def edge_rate(step: object) -> str:
    return f"{clamp_rate_step(step) * 6:+d}%"


def clamp_volume(value: object, fallback: float = 1.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError, OverflowError):
        return fallback
