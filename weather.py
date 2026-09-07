"""Small Open-Meteo weather helper used by the voice assistant.

The service does not require an API key. Network failures are intentionally
returned as a readable context string so ordinary chat can still continue.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime

import config


WEATHER_WORDS = re.compile(
    r"天气|气温|温度|下雨|降雨|下雪|天气预报|forecast|weather|temperature",
    re.IGNORECASE,
)
KNOWN_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "重庆",
    "武汉",
    "西安",
    "天津",
    "青岛",
    "厦门",
    "郑州",
    "长沙",
    "昆明",
    "沈阳",
    "哈尔滨",
    "福州",
    "济南",
    "合肥",
    "大连",
    "宁波",
    "东莞",
    "佛山",
    "香港",
    "澳门",
    "台北",
)
_CITY_RE = re.compile(r"(?:在|到|去|查一下|查询)?\s*([\u4e00-\u9fff]{2,12})(?:市|县)?\s*(?:的)?\s*(?:天气|气温|温度)")
_EN_CITY_RE = re.compile(r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]{1,40})", re.IGNORECASE)
_CACHE: dict[str, tuple[float, str]] = {}


def _http_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenAIQ-Voice-Python/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_city(text: str) -> str | None:
    """Extract a likely city without trying to infer the user's GPS location."""
    for city in KNOWN_CITIES:
        if city in text:
            return city
    match = _CITY_RE.search(text)
    if match:
        return match.group(1).strip()
    matches = _EN_CITY_RE.findall(text)
    if matches:
        city = matches[-1].strip()
        city = re.sub(r"\s+(?:weather|forecast|temperature)\s*$", "", city, flags=re.IGNORECASE)
        if city:
            return city
    return config.WEATHER_DEFAULT_CITY.strip() or None


def is_weather_query(text: str) -> bool:
    return bool(WEATHER_WORDS.search(text))


def _weather_description(code: int | None) -> str:
    codes = {
        0: "晴",
        1: "大部晴朗",
        2: "局部多云",
        3: "阴",
        45: "有雾",
        48: "有雾凇",
        51: "小毛毛雨",
        53: "毛毛雨",
        55: "较强毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "阵雨",
        81: "较强阵雨",
        82: "强阵雨",
        95: "雷雨",
        96: "雷雨并伴有冰雹",
        99: "雷雨并伴有较强冰雹",
    }
    return codes.get(code, "天气情况未知")


def lookup_weather(text: str, timeout: float | None = None) -> str | None:
    """Return a model-ready weather context, or None for ordinary questions."""
    if not is_weather_query(text):
        return None
    city = extract_city(text)
    if not city:
        return "实时天气工具未查询：用户没有提供城市，不能根据语音自动确定所在位置。请先询问用户所在城市。"
    timeout = timeout if timeout is not None else config.WEATHER_TIMEOUT_SECONDS
    try:
        cache_key = city.casefold()
        now = datetime.now().timestamp()
        cached = _CACHE.get(cache_key)
        if cached and now - cached[0] < config.WEATHER_CACHE_SECONDS:
            return cached[1]

        query = urllib.parse.urlencode({"name": city, "count": 1, "language": "zh", "format": "json"})
        geo = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}", timeout)
        results = geo.get("results") or []
        if not results:
            return f"实时天气工具查询失败：找不到城市“{city}”。"
        place = results[0]
        latitude, longitude = place["latitude"], place["longitude"]
        forecast_query = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 3,
                "timezone": "auto",
            }
        )
        forecast = _http_json(f"https://api.open-meteo.com/v1/forecast?{forecast_query}", timeout)
        current = forecast.get("current", {})
        daily = forecast.get("daily", {})
        current_text = (
            f"当前{_weather_description(current.get('weather_code'))}，"
            f"气温 {current.get('temperature_2m', '?')}°C，体感 {current.get('apparent_temperature', '?')}°C，"
            f"湿度 {current.get('relative_humidity_2m', '?')}%，风速 {current.get('wind_speed_10m', '?')} km/h。"
        )
        days: list[str] = []
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_probability_max", [])
        labels = ["今天", "明天", "后天"]
        for i, date in enumerate(dates[:3]):
            label = labels[i] if i < len(labels) else date
            days.append(
                f"{label}（{date}）：{_weather_description(codes[i] if i < len(codes) else None)}，"
                f"{lows[i] if i < len(lows) else '?'}~{highs[i] if i < len(highs) else '?'}°C，"
                f"降水概率 {rain[i] if i < len(rain) else '?'}%。"
            )
        context = (
            f"实时天气工具结果（地点：{place.get('name', city)}，来源：Open-Meteo，查询时间：{datetime.now():%Y-%m-%d %H:%M}）："
            + current_text
            + " "
            + " ".join(days)
            + " 请基于这些数据回答，不要声称自己能自动获取用户定位。"
        )
        _CACHE[cache_key] = (now, context)
        return context
    except Exception as exc:
        return f"实时天气工具暂时不可用（{type(exc).__name__}）。请如实告知用户无法获取实时天气，不要编造天气数据。"
