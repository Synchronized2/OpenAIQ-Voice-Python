from __future__ import annotations

import ctypes
import asyncio
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config


@dataclass(frozen=True)
class LocalAction:
    kind: str
    value: int | str | None = None


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    message: str


_TRAILING_PUNCTUATION = " \t\r\n。.!！?？,，;；"
_EXIT_PHRASES = {
    "拜拜",
    "再见",
    "退出语音",
    "结束语音",
    "结束对话",
    "停止聆听",
    "先这样",
    "先这样吧",
    "byebye",
    "goodbye",
    "exitvoice",
    "stoplistening",
}


def _compact(text: str) -> str:
    return re.sub(r"[\s，,。.!！?？;；:：]", "", text.casefold())


def is_voice_exit_phrase(text: str) -> bool:
    return _compact(text) in _EXIT_PHRASES


def _clean(text: str) -> str:
    return text.strip(_TRAILING_PUNCTUATION)


def _parse_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "一百":
        return 100
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(char in digits for char in value):
        return int("".join(str(digits[char]) for char in value))
    return None


def _parse_percent(match: re.Match[str]) -> int | None:
    value = _parse_number(match.group("value"))
    if value is None:
        return None
    return value if 0 <= value <= 100 else None


def _match_exact_percent(text: str, subject: str) -> int | None:
    if subject == "volume":
        chinese_subject = r"(?:系统|电脑|主)?音量"
        english_subject = r"(?:(?:system|master)\s+)?volume"
    else:
        chinese_subject = r"(?:屏幕|显示器|电脑)?亮度"
        english_subject = r"(?:(?:screen|display)\s+)?brightness"

    patterns = (
        rf"^(?:请)?(?:帮我)?(?:把|将)?{chinese_subject}(?:调到|调整到|设置为|设为|调成|设置到|调高到|提高到|增大到|加大到|降低到|调低到|减小到|调暗到|调亮到|至)(?:百分之)?(?P<value>\d{{1,3}}|[零〇一二两三四五六七八九十百]+)(?:%|％)?$",
        rf"^(?:请)?(?:帮我)?(?:把|将)?{chinese_subject}(?:调|设置|提高|降低|调高|调低)?(?:到|至|为)(?:百分之)?(?P<value>\d{{1,3}}|[零〇一二两三四五六七八九十百]+)(?:%|％)?$",
        rf"^(?:please\s+)?(?:set|change|turn)\s+(?:the\s+)?{english_subject}(?:\s+(?:to|at))?\s*(?P<value>\d{{1,3}})\s*%?$",
        rf"^{english_subject}\s*(?:to|at|=)\s*(?P<value>\d{{1,3}})\s*%?$",
    )
    for pattern in patterns:
        for candidate in (text, re.sub(r"\s+", "", text)):
            match = re.fullmatch(pattern, candidate, re.IGNORECASE)
            if match:
                return _parse_percent(match)
    return None


def _change_amount(text: str, default: int) -> int:
    match = re.search(r"(\d{1,2})\s*(?:%|％)?", text)
    return min(100, int(match.group(1))) if match else default


def parse_local_intent(text: str) -> LocalAction | None:
    """Return only high-confidence, explicitly requested local actions."""
    if not getattr(config, "AGENT_ENABLED", True):
        return None
    cleaned = _clean(text)
    compact = _compact(cleaned)
    if not compact:
        return None

    volume = _match_exact_percent(cleaned, "volume")
    if volume is not None:
        return LocalAction("set_volume", volume)
    brightness = _match_exact_percent(cleaned, "brightness")
    if brightness is not None:
        return LocalAction("set_brightness", brightness)

    if compact in {"静音", "设为静音", "设置静音", "系统静音", "mute", "mutevolume"}:
        return LocalAction("mute", True)
    if compact in {"取消静音", "解除静音", "恢复声音", "unmute", "unmutevolume"}:
        return LocalAction("mute", False)

    step = int(getattr(config, "AGENT_ADJUST_STEP", 10))
    volume_up = (
        r"^(?:请)?(?:帮我)?(?:把|将)?(?:系统|电脑)?音量(?:调高|提高|增大|加大|增加|大)(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:请)?(?:帮我)?(?:调高|提高|增大|加大|增加)(?:系统|电脑)?音量(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:please\s+)?(?:turn|raise|increase)\s+(?:the\s+)?volume(?:\s+(?:by\s+)?\d{1,2}%?)?$",
        r"^volume\s+up(?:\s+\d{1,2}%?)?$",
    )
    volume_down = (
        r"^(?:请)?(?:帮我)?(?:把|将)?(?:系统|电脑)?音量(?:调低|降低|减小|调小|小|减少)(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:请)?(?:帮我)?(?:调低|降低|减小|调小|减少)(?:系统|电脑)?音量(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:please\s+)?(?:turn|lower|decrease|reduce)\s+(?:the\s+)?volume(?:\s+(?:by\s+)?\d{1,2}%?)?$",
        r"^volume\s+down(?:\s+\d{1,2}%?)?$",
    )
    compact_cleaned = re.sub(r"\s+", "", cleaned)
    if any(
        re.fullmatch(pattern, candidate, re.IGNORECASE)
        for pattern in volume_up
        for candidate in (cleaned, compact_cleaned)
    ):
        return LocalAction("change_volume", _change_amount(cleaned, step))
    if any(
        re.fullmatch(pattern, candidate, re.IGNORECASE)
        for pattern in volume_down
        for candidate in (cleaned, compact_cleaned)
    ):
        return LocalAction("change_volume", -_change_amount(cleaned, step))

    brightness_up = (
        r"^(?:请)?(?:帮我)?(?:把|将)?(?:屏幕|显示器|电脑)?亮度(?:调高|提高|增大|增加|调亮|亮)(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:请)?(?:帮我)?(?:调高|提高|增大|增加|调亮)(?:屏幕|显示器|电脑)?亮度(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:please\s+)?(?:turn|raise|increase)\s+(?:the\s+)?brightness(?:\s+(?:by\s+)?\d{1,2}%?)?$",
        r"^brightness\s+up(?:\s+\d{1,2}%?)?$",
    )
    brightness_down = (
        r"^(?:请)?(?:帮我)?(?:把|将)?(?:屏幕|显示器|电脑)?亮度(?:调低|降低|减小|减少|调暗|暗)(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:请)?(?:帮我)?(?:调低|降低|减小|减少|调暗)(?:屏幕|显示器|电脑)?亮度(?:一点)?(?:\d{1,2}(?:%|％)?)?$",
        r"^(?:please\s+)?(?:turn|lower|decrease|reduce)\s+(?:the\s+)?brightness(?:\s+(?:by\s+)?\d{1,2}%?)?$",
        r"^brightness\s+down(?:\s+\d{1,2}%?)?$",
    )
    if any(
        re.fullmatch(pattern, candidate, re.IGNORECASE)
        for pattern in brightness_up
        for candidate in (cleaned, compact_cleaned)
    ):
        return LocalAction("change_brightness", _change_amount(cleaned, step))
    if any(
        re.fullmatch(pattern, candidate, re.IGNORECASE)
        for pattern in brightness_down
        for candidate in (cleaned, compact_cleaned)
    ):
        return LocalAction("change_brightness", -_change_amount(cleaned, step))

    media_commands = {
        "播放": "play",
        "播放音乐": "play",
        "继续播放": "play",
        "暂停": "pause",
        "暂停播放": "pause",
        "暂停音乐": "pause",
        "下一首": "next",
        "切到下一首": "next",
        "上一首": "previous",
        "切到上一首": "previous",
        "停止播放": "stop",
        "play": "play",
        "playmusic": "play",
        "resumeplayback": "play",
        "pause": "pause",
        "pausemusic": "pause",
        "next": "next",
        "nexttrack": "next",
        "previoustrack": "previous",
        "stopmusic": "stop",
    }
    if compact in media_commands:
        return LocalAction("media", media_commands[compact])

    if compact in {"锁屏", "锁定电脑", "锁定屏幕", "lockcomputer", "lockscreen"}:
        return LocalAction("lock")
    if compact in {"显示桌面", "回到桌面", "showdesktop"}:
        return LocalAction("show_desktop")

    open_match = re.fullmatch(
        r"(?:请)?(?:帮我)?(?:打开|启动|运行)\s*(?P<target>.+?)",
        cleaned,
        re.IGNORECASE,
    )
    if not open_match:
        open_match = re.fullmatch(
            r"(?:please\s+)?(?:open|launch|start)(?:\s+the)?\s+(?P<target>.+?)(?:\s+app(?:lication)?)?",
            cleaned,
            re.IGNORECASE,
        )
    if open_match:
        target = open_match.group("target").strip(_TRAILING_PUNCTUATION)
        if 0 < len(target) <= 40 and not re.search(r"[\\/:;|&<>]", target):
            return LocalAction("open_application", target)
    return None


class LocalAgent:
    _MEDIA_KEYS = {
        "play": 0xB3,
        "pause": 0xB3,
        "next": 0xB0,
        "previous": 0xB1,
        "stop": 0xB2,
    }

    _SYSTEM_TARGETS = {
        "设置": ("Windows 设置", "ms-settings:"),
        "系统设置": ("Windows 设置", "ms-settings:"),
        "windowssettings": ("Windows 设置", "ms-settings:"),
        "蓝牙设置": ("蓝牙设置", "ms-settings:bluetooth"),
        "bluetoothsettings": ("蓝牙设置", "ms-settings:bluetooth"),
        "wifi设置": ("网络设置", "ms-settings:network-wifi"),
        "wlan设置": ("网络设置", "ms-settings:network-wifi"),
        "networksettings": ("网络设置", "ms-settings:network"),
        "声音设置": ("声音设置", "ms-settings:sound"),
        "音量设置": ("声音设置", "ms-settings:sound"),
        "soundsettings": ("声音设置", "ms-settings:sound"),
        "显示设置": ("显示设置", "ms-settings:display"),
        "亮度设置": ("显示设置", "ms-settings:display"),
        "displaysettings": ("显示设置", "ms-settings:display"),
        "相机": ("相机", "microsoft.windows.camera:"),
        "camera": ("相机", "microsoft.windows.camera:"),
        "记事本": ("记事本", "notepad.exe"),
        "notepad": ("记事本", "notepad.exe"),
        "计算器": ("计算器", "calc.exe"),
        "calculator": ("计算器", "calc.exe"),
        "画图": ("画图", "mspaint.exe"),
        "paint": ("画图", "mspaint.exe"),
        "文件管理器": ("文件资源管理器", "explorer.exe"),
        "资源管理器": ("文件资源管理器", "explorer.exe"),
        "fileexplorer": ("文件资源管理器", "explorer.exe"),
        "任务管理器": ("任务管理器", "taskmgr.exe"),
        "taskmanager": ("任务管理器", "taskmgr.exe"),
        "终端": ("Windows 终端", "wt.exe"),
        "windowsterminal": ("Windows 终端", "wt.exe"),
        "命令提示符": ("命令提示符", "cmd.exe"),
        "cmd": ("命令提示符", "cmd.exe"),
        "下载": ("下载", "shell:Downloads"),
        "下载文件夹": ("下载", "shell:Downloads"),
        "downloads": ("下载", "shell:Downloads"),
        "文档": ("文档", "shell:Personal"),
        "documents": ("文档", "shell:Personal"),
    }

    _APP_SEARCH_ALIASES = {
        "qq音乐": ("QQ 音乐", ("QQ音乐", "QQMusic")),
        "qqmusic": ("QQ 音乐", ("QQ音乐", "QQMusic")),
        "qq": ("QQ", ("QQ",)),
        "微信": ("微信", ("微信", "WeChat")),
        "wechat": ("微信", ("微信", "WeChat")),
        "网易云音乐": ("网易云音乐", ("网易云音乐", "NetEase CloudMusic")),
        "cloudmusic": ("网易云音乐", ("网易云音乐", "NetEase CloudMusic")),
        "chrome": ("Google Chrome", ("Google Chrome", "Chrome")),
        "谷歌浏览器": ("Google Chrome", ("Google Chrome", "Chrome")),
        "edge": ("Microsoft Edge", ("Microsoft Edge",)),
        "vscode": ("Visual Studio Code", ("Visual Studio Code",)),
        "visualstudiocode": ("Visual Studio Code", ("Visual Studio Code",)),
        "spotify": ("Spotify", ("Spotify",)),
    }

    def execute(self, action: LocalAction) -> AgentResult:
        try:
            if action.kind == "set_volume":
                return self._set_volume(int(action.value))
            if action.kind == "change_volume":
                return self._change_volume(int(action.value))
            if action.kind == "mute":
                return self._set_mute(bool(action.value))
            if action.kind == "set_brightness":
                return self._set_brightness(int(action.value))
            if action.kind == "change_brightness":
                return self._change_brightness(int(action.value))
            if action.kind == "open_application":
                return self._open_application(str(action.value))
            if action.kind == "media":
                return self._media(str(action.value))
            if action.kind == "lock":
                ctypes.windll.user32.LockWorkStation()
                return AgentResult(True, "已锁定电脑。")
            if action.kind == "show_desktop":
                self._press_hotkey(0x5B, 0x44)
                return AgentResult(True, "已显示桌面。")
            return AgentResult(False, "这个本地操作暂不支持。")
        except Exception as exc:
            return AgentResult(False, f"本地操作失败：{self._friendly_error(exc)}")

    def _set_volume(self, percent: int) -> AgentResult:
        with self._audio_endpoint() as endpoint:
            endpoint.SetMasterVolumeLevelScalar(max(0, min(100, percent)) / 100.0, None)
            actual = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        return AgentResult(True, f"已将音量调到 {actual}%。")

    def _change_volume(self, delta: int) -> AgentResult:
        with self._audio_endpoint() as endpoint:
            current = round(endpoint.GetMasterVolumeLevelScalar() * 100)
            target = max(0, min(100, current + delta))
            endpoint.SetMasterVolumeLevelScalar(target / 100.0, None)
            actual = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        return AgentResult(True, f"已将音量调到 {actual}%。")

    def _set_mute(self, muted: bool) -> AgentResult:
        with self._audio_endpoint() as endpoint:
            endpoint.SetMute(1 if muted else 0, None)
        return AgentResult(True, "已静音。" if muted else "已取消静音。")

    @staticmethod
    @contextmanager
    def _audio_endpoint():
        try:
            import comtypes
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        except ImportError as exc:
            raise RuntimeError("缺少 pycaw，请重新运行 setup.ps1 安装依赖") from exc
        comtypes.CoInitialize()
        endpoint = None
        device = None
        try:
            device = AudioUtilities.GetSpeakers()
            endpoint = getattr(device, "EndpointVolume", None)
            if endpoint is None:
                from comtypes import CLSCTX_ALL
                from ctypes import POINTER, cast

                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            yield endpoint
        finally:
            endpoint = None
            device = None
            comtypes.CoUninitialize()

    def _set_brightness(self, percent: int) -> AgentResult:
        value = max(0, min(100, percent))
        try:
            import screen_brightness_control as sbc

            sbc.set_brightness(value)
            actual_values = sbc.get_brightness()
            actual = round(sum(actual_values) / len(actual_values)) if actual_values else value
            return AgentResult(True, f"已将屏幕亮度调到 {actual}%。")
        except ImportError:
            script = (
                "$m=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods;"
                "if($null -eq $m){throw 'NO_SUPPORTED_DISPLAY'};"
                f"$m|Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=0;Brightness=[byte]{value}}}|Out-Null"
            )
            self._run_powershell(script)
            return AgentResult(True, f"已将屏幕亮度调到 {value}%。")
        except Exception as exc:
            raise RuntimeError("当前显示器不支持软件亮度调节") from exc

    def _change_brightness(self, delta: int) -> AgentResult:
        try:
            import screen_brightness_control as sbc

            values = sbc.get_brightness()
            if not values:
                raise RuntimeError("无法读取当前亮度")
            current = round(sum(values) / len(values))
        except ImportError:
            script = (
                "$b=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness|"
                "Select-Object -First 1 -ExpandProperty CurrentBrightness;"
                "if($null -eq $b){throw 'NO_SUPPORTED_DISPLAY'};Write-Output $b"
            )
            output = self._run_powershell(script)
            try:
                current = int(output.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise RuntimeError("无法读取当前亮度") from exc
        except Exception as exc:
            raise RuntimeError("当前显示器不支持软件亮度调节") from exc
        return self._set_brightness(current + delta)

    @staticmethod
    def _run_powershell(script: str) -> str:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            creationflags=flags,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            if "NO_SUPPORTED_DISPLAY" in detail:
                raise RuntimeError("当前显示器不支持 Windows WMI 亮度调节")
            raise RuntimeError(detail or "Windows 拒绝了亮度操作")
        return result.stdout

    def _open_application(self, requested: str) -> AgentResult:
        key = self._normalize_app_name(requested)
        custom = getattr(config, "AGENT_APPLICATIONS", {})
        for alias, target in custom.items():
            if self._normalize_app_name(str(alias)) == key:
                return self._launch(str(alias), os.path.expandvars(str(target)))

        if key in self._SYSTEM_TARGETS:
            display, target = self._SYSTEM_TARGETS[key]
            return self._launch(display, target)

        display = requested.strip()
        search_names = (display,)
        if key in self._APP_SEARCH_ALIASES:
            display, search_names = self._APP_SEARCH_ALIASES[key]
        shortcut = self._find_start_menu_shortcut(search_names)
        if shortcut:
            return self._launch(display, str(shortcut))

        executable = shutil.which(requested.strip())
        if executable:
            return self._launch(display, executable)
        return AgentResult(False, f"未找到“{display}”。请使用开始菜单中的完整名称，或在 config.py 添加应用别名。")

    @staticmethod
    def _launch(display: str, target: str) -> AgentResult:
        if not target:
            return AgentResult(False, f"“{display}”没有配置启动路径。")
        if ("\\" in target or "/" in target) and not Path(target).exists():
            return AgentResult(False, f"“{display}”的启动路径不存在。")
        os.startfile(target)  # type: ignore[attr-defined]
        return AgentResult(True, f"已打开“{display}”。")

    @classmethod
    @lru_cache(maxsize=128)
    def _find_start_menu_shortcut(cls, search_names: tuple[str, ...]) -> Path | None:
        roots = (
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        )
        shortcuts: list[Path] = []
        for root in roots:
            if root.is_dir():
                shortcuts.extend(root.rglob("*.lnk"))
        normalized_names = {cls._normalize_app_name(name) for name in search_names}
        exact = [path for path in shortcuts if cls._normalize_app_name(path.stem) in normalized_names]
        if len(exact) == 1:
            return exact[0]
        partial = [
            path
            for path in shortcuts
            if any(
                len(name) >= 3 and name in cls._normalize_app_name(path.stem)
                for name in normalized_names
            )
        ]
        return partial[0] if len(partial) == 1 else None

    @staticmethod
    def _normalize_app_name(value: str) -> str:
        normalized = re.sub(r"[\s._-]", "", value.casefold())
        for suffix in ("application", "客户端", "应用程序", "软件", "应用", "app"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        return normalized

    def _media(self, command: str) -> AgentResult:
        if command in {"play", "pause"}:
            return asyncio.run(self._set_media_playback(command))
        key = self._MEDIA_KEYS[command]
        self._press_key(key)
        messages = {
            "play": "已发送播放命令。",
            "pause": "已发送暂停命令。",
            "next": "已发送下一首命令。",
            "previous": "已发送上一首命令。",
            "stop": "已发送停止播放命令。",
        }
        return AgentResult(True, messages[command])

    @staticmethod
    async def _set_media_playback(command: str) -> AgentResult:
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        )

        async def change():
            manager = await SessionManager.request_async()
            session = manager.get_current_session()
            if session is None:
                return AgentResult(False, "没有可控制的媒体，请先打开音乐或视频应用。")
            operation = session.try_play_async if command == "play" else session.try_pause_async
            if not await operation():
                return AgentResult(False, "当前媒体应用不支持这项操作。")
            return AgentResult(True, "已播放。" if command == "play" else "已暂停。")

        try:
            return await asyncio.wait_for(change(), timeout=5.0)
        except TimeoutError:
            return AgentResult(False, "媒体应用响应超时。")

    @staticmethod
    def _press_key(key: int) -> None:
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)
        ctypes.windll.user32.keybd_event(key, 0, 0x0002, 0)

    @classmethod
    def _press_hotkey(cls, modifier: int, key: int) -> None:
        ctypes.windll.user32.keybd_event(modifier, 0, 0, 0)
        cls._press_key(key)
        ctypes.windll.user32.keybd_event(modifier, 0, 0x0002, 0)

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        text = str(exc).strip()
        if isinstance(exc, subprocess.TimeoutExpired):
            return "系统操作超时"
        return text[:240] or exc.__class__.__name__


def handle_local_action(text: str, agent: LocalAgent | None = None) -> AgentResult | None:
    action = parse_local_intent(text)
    return None if action is None else (agent or LocalAgent()).execute(action)
