from __future__ import annotations

from typing import Any

import httpx
from openai import OpenAI


IMAGE_MARKERS = ("image", "dall-e", "dalle", "flux", "stable-diffusion", "ideogram")
NON_CHAT_MARKERS = (
    *IMAGE_MARKERS,
    "embedding",
    "moderation",
    "whisper",
    "transcribe",
    "speech",
    "tts",
    "realtime",
)


class ModelCatalogError(RuntimeError):
    pass


def classify_models(model_ids: list[str]) -> tuple[list[str], list[str]]:
    unique = sorted({model.strip() for model in model_ids if model and model.strip()}, key=str.lower)
    image = [model for model in unique if any(mark in model.lower() for mark in IMAGE_MARKERS)]
    chat = [model for model in unique if not any(mark in model.lower() for mark in NON_CHAT_MARKERS)]
    return chat, image


def fetch_model_ids(base_url: str, api_key: str, *, client: Any | None = None) -> list[str]:
    clean_url = base_url.strip()
    clean_key = api_key.strip()
    if not clean_url:
        raise ModelCatalogError("服务 URL 不能为空。")
    if not clean_key:
        raise ModelCatalogError("API Key 不能为空。")
    owns_client = client is None
    try:
        if client is None:
            client = OpenAI(
                api_key=clean_key,
                base_url=clean_url,
                timeout=httpx.Timeout(20.0, connect=10.0),
                max_retries=0,
            )
        response = client.models.list()
        ids = [str(item.id).strip() for item in getattr(response, "data", []) if getattr(item, "id", None)]
        if not ids:
            raise ModelCatalogError("服务没有返回任何模型。你仍然可以手动输入模型名称。")
        return sorted(set(ids), key=str.lower)
    except ModelCatalogError:
        raise
    except Exception as exc:
        message = str(exc).replace(clean_key, "[REDACTED]")
        raise ModelCatalogError(f"获取模型列表失败：{message}") from exc
    finally:
        if owns_client and client is not None:
            client.close()
