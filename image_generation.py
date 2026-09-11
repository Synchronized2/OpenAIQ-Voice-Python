from __future__ import annotations

import asyncio
import base64
import binascii
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from openai import AsyncOpenAI

from cancellation import OperationCancelled, run_cancellable
from diagnostics import log


MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
IMAGE_GENERATION_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "当用户想创建、生成、绘制一张新图片、照片、插画、海报、壁纸或头像时调用。"
            "不要在用户只是询问图片、绘画技巧或生图方法时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "完整、具体的中文生图提示词，保留用户要求的主体、风格、构图和细节。",
                }
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
}
IMAGE_TOOL_GUIDANCE = (
    "你拥有一个可用的外部工具 generate_image，它由独立的生图模型执行，"
    "不要求当前对话模型自身具备图像生成能力。"
    "当用户表达想创建、生成、绘制新图片、照片、插画、海报、壁纸或头像的意图时，"
    "必须调用 generate_image，并把用户的主体、风格、构图与细节整理进 prompt。"
    "不要声称当前模型无法生图，也不要只返回提示词。"
    "用户只是讨论已有图片、询问绘画技巧或生图方法时不要调用工具，正常回答即可。"
)


class ImageGenerationError(RuntimeError):
    """A user-facing image error that must not expose credentials."""


class ImageGenerator:
    def __init__(self, *, base_url: str, api_key: str, model: str, output_dir: str | Path) -> None:
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.output_dir = Path(output_dir).expanduser()
        if not self.base_url:
            raise ImageGenerationError("生图服务 URL 不能为空。")
        if not self.api_key:
            raise ImageGenerationError("生图服务 API Key 不能为空。")
        if not self.model:
            raise ImageGenerationError("生图模型不能为空。")

    def generate(self, prompt: str, *, should_stop=None) -> Path:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ImageGenerationError("图片描述不能为空。")
        stopped = should_stop or (lambda: False)
        try:
            return run_cancellable(self._generate(clean_prompt), stopped)
        except OperationCancelled:
            raise
        except ImageGenerationError:
            raise
        except BaseException as exc:
            message = str(exc).replace(self.api_key, "[REDACTED]")
            raise ImageGenerationError(f"图片生成失败：{message}") from exc

    async def _generate(self, prompt: str) -> Path:
        started = asyncio.get_running_loop().time()
        log.info("image.request_start model=%s", self.model)
        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(180.0, connect=15.0),
            max_retries=1,
        ) as client:
            response = await client.images.generate(
                model=self.model,
                prompt=prompt,
                response_format="b64_json",
            )
        content = await self._extract_image_bytes(response)
        extension = self._detect_extension(content)
        output_path = self._new_output_path(extension)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as image_file:
            image_file.write(content)
        log.info(
            "image.request_end model=%s seconds=%.3f bytes=%s output=%s",
            self.model,
            asyncio.get_running_loop().time() - started,
            len(content),
            output_path.name,
        )
        return output_path.resolve()

    async def _extract_image_bytes(self, response: Any) -> bytes:
        data = getattr(response, "data", None)
        if not data:
            raise ImageGenerationError("服务未返回图片数据。")
        item = data[0]
        encoded = getattr(item, "b64_json", None)
        if encoded:
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ImageGenerationError("服务返回的 Base64 图片数据无效。") from exc
        image_url = getattr(item, "url", None)
        if image_url:
            return await self._download_image(image_url)
        raise ImageGenerationError("服务响应中没有 b64_json 或 url 图片字段。")

    async def _download_image(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ImageGenerationError("服务返回了不安全的图片下载地址。")
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    size = response.headers.get("content-length")
                    if size and int(size) > MAX_DOWNLOAD_BYTES:
                        raise ImageGenerationError("服务返回的图片超过 50 MB 限制。")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ImageGenerationError("服务返回的图片超过 50 MB 限制。")
                        chunks.append(chunk)
                    return b"".join(chunks)
        except ImageGenerationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ImageGenerationError(f"下载生成图片失败：{exc}") from exc

    @staticmethod
    def _detect_extension(content: bytes) -> str:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if content.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
            return ".webp"
        raise ImageGenerationError("服务返回的数据不是受支持的 PNG、JPEG 或 WebP 图片。")

    def _new_output_path(self, extension: str) -> Path:
        output_dir = self.output_dir
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return output_dir / f"image-{stamp}-{uuid4().hex[:8]}{extension}"
