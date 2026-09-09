"""Safe defaults. setup.ps1 creates your private, Git-ignored config.py."""

import os
from pathlib import Path

CHAT_BASE_URL = "https://www.pcie.cloud/v1"
CHAT_API_KEY = ""
CHAT_MODEL = "gpt-5.3-codex-spark"
CHAT_REASONING_EFFORT = "low"
CHAT_FIRST_TOKEN_TIMEOUT_SECONDS = 30.0
CHAT_TOTAL_TIMEOUT_SECONDS = 90.0
CHAT_MODEL_OPTIONS = (
    ("gpt-5.3-codex-spark", "低延迟语音对话"),
    ("gpt-5.4-mini", "轻量对话"),
    ("gpt-5.6-luna", "语音对话"),
    ("gpt-5.6-terra", "质量和速度平衡"),
    ("gpt-5.6-sol", "复杂问题和代码"),
    ("qwen3.7-plus", "中文对话"),
)

TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TTS_RATE = "+0%"
TTS_VOLUME = 1.0
TTS_RETRIES = 3
TTS_RETRY_DELAY_SECONDS = 1.0
TTS_SEGMENT_MIN_CHARS = 24
TTS_SEGMENT_MAX_CHARS = 100
TTS_BARGE_IN_ENABLED = True
SYSTEM_PROMPT = (
    "你是一个可靠、自然、有温度的中文语音陪伴助手。"
    "回答应适合直接朗读，简洁但不敷衍，不要使用表格或复杂 Markdown。"
)
TEMPERATURE = 0.7
MAX_TOKENS = 1200
HISTORY_TURNS = 10
WEATHER_DEFAULT_CITY = "北京"
WEATHER_TIMEOUT_SECONDS = 6.0
WEATHER_CACHE_SECONDS = 300.0

# setup.ps1 downloads verified model files into this project.
ASR_MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "paraformer-large")
ASR_VAD_MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "FireRedVAD")
ASR_THREADS = 8
# None selects this computer's default microphone, not a different PC's index.
ASR_DEVICE: int | None = None
ASR_VAD_THRESHOLD = 0.4
ASR_VAD_END_SILENCE = 0.7
ASR_PRE_ROLL = 0.4
ASR_POST_ROLL = 0.26
ASR_HOTWORDS = "孙悟空 猴哥 悟空 大圣 齐天大圣 QQ音乐 OpenAIQ 音量 亮度 相机 微信"

UI_CORE_STYLE = "PET"
PET_SPRITESHEET = str(Path(__file__).resolve().parent / "assets" / "super-goku" / "spritesheet.png")
PET_IDLE_ROW = 0
PET_FRAME_COUNT = 8
COMPACT_PET_SIZE = 138
UI_ANIMATION_FPS = 22
PET_FRAME_INTERVAL_MS = 150
LIVE2D_MODEL_DIR = str(Path(__file__).resolve().parent / "assets" / "live2d")
WAKE_WORD_ENABLED = True
WAKE_WORDS = ("孙悟空", "猴哥", "悟空", "大圣", "齐天大圣")
WAKE_WORD_ALIASES = ("孙悟", "五空", "吾空", "武空", "后哥", "大胜", "齐天大胜")
WAKE_COOLDOWN_SECONDS = 4.0
WAKE_MIN_CHARS = 2
WAKE_MAX_CHARS = 12
AGENT_ENABLED = True
AGENT_ADJUST_STEP = 10
AGENT_APPLICATIONS = {}


def chat_api_key() -> str:
    return CHAT_API_KEY.strip() or os.getenv("PCIE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
