from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import config
from chat import ChatSession
from image_generation import ImageGenerationError, ImageGenerator
from model_catalog import ModelCatalogError, classify_models, fetch_model_ids


PNG = b"\x89PNG\r\n\x1a\nimage"


class FakeToolStream:
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        function = lambda name, arguments: SimpleNamespace(name=name, arguments=arguments)
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(index=0, id="call-1", function=function("generate_", '{"prompt":"月球'))],
        ))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(index=0, id=None, function=function("image", '上的孙悟空"}'))],
        ))])

    async def aclose(self):
        self.closed = True


class ChatImageToolTests(unittest.TestCase):
    def test_dialog_model_tool_call_drives_image_request(self):
        stream = FakeToolStream()
        chat = ChatSession.__new__(ChatSession)
        chat.messages = [{"role": "system", "content": "test"}]
        chat._stream = lambda: stream
        with patch("chat.lookup_weather", return_value=None):
            answer, interrupted = chat.reply("我想生成一张照片")
        self.assertEqual(answer, "")
        self.assertFalse(interrupted)
        self.assertEqual(chat.last_tool_call["prompt"], "月球上的孙悟空")
        self.assertTrue(stream.closed)
        self.assertEqual(chat.messages[-1]["role"], "user")
        chat.record_image_result(chat.last_tool_call["prompt"], "result.png")
        self.assertIn("result.png", chat.messages[-1]["content"])

    def test_spark_image_request_uses_tool_capable_model_preflight(self):
        chat = ChatSession.__new__(ChatSession)
        chat.model = "gpt-5.3-codex-spark"
        chat.messages = [{"role": "system", "content": "test"}]
        expected = {
            "id": "router-1",
            "name": "generate_image",
            "prompt": "孙悟空和二郎神大战，电影感",
        }
        chat._preflight_image_tool = Mock(return_value=expected)
        chat._stream = Mock(side_effect=AssertionError("Spark should not answer first"))

        with patch("chat.lookup_weather", return_value=None):
            answer, interrupted = chat.reply("给我生成一张孙悟空和二郎神大战的图片")

        self.assertEqual(answer, "")
        self.assertFalse(interrupted)
        self.assertEqual(chat.last_tool_call, expected)
        self.assertEqual(chat.messages[-1]["role"], "user")
        chat._stream.assert_not_called()

    def test_spark_preflight_gate_does_not_directly_generate_images(self):
        chat = ChatSession.__new__(ChatSession)
        chat.model = "gpt-5.3-codex-spark"
        self.assertTrue(chat._needs_image_tool_preflight("帮我生成一张月球照片"))
        self.assertTrue(chat._needs_image_tool_preflight("这张图片是怎么生成的？"))
        self.assertFalse(chat._needs_image_tool_preflight("今天晚上吃什么？"))
        chat.model = "gpt-5.6-luna"
        self.assertFalse(chat._needs_image_tool_preflight("帮我生成一张月球照片"))


class ImageGeneratorTests(unittest.TestCase):
    def test_base64_image_is_saved_inside_output_directory(self):
        item = SimpleNamespace(b64_json=base64.b64encode(PNG).decode(), url=None)
        response = SimpleNamespace(data=[item])
        images = SimpleNamespace(generate=AsyncMock(return_value=response))
        client = AsyncMock()
        client.images = images
        client.__aenter__.return_value = client
        with tempfile.TemporaryDirectory() as directory, patch("image_generation.AsyncOpenAI", return_value=client):
            generator = ImageGenerator(base_url="https://example.com/v1", api_key="secret", model="gpt-image-2", output_dir=directory)
            saved = generator.generate("星空城市")
            self.assertEqual(saved.read_bytes(), PNG)
            self.assertEqual(saved.parent, Path(directory).resolve())
            images.generate.assert_awaited_once_with(model="gpt-image-2", prompt="星空城市", response_format="b64_json")

    def test_invalid_image_is_not_written(self):
        item = SimpleNamespace(b64_json=base64.b64encode(b"not-image").decode(), url=None)
        client = AsyncMock()
        client.images = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(data=[item])))
        client.__aenter__.return_value = client
        with tempfile.TemporaryDirectory() as directory, patch("image_generation.AsyncOpenAI", return_value=client):
            generator = ImageGenerator(base_url="https://example.com/v1", api_key="secret", model="image", output_dir=directory)
            with self.assertRaisesRegex(ImageGenerationError, "PNG"):
                generator.generate("test")
            self.assertEqual(list(Path(directory).iterdir()), [])


class ModelCatalogTests(unittest.TestCase):
    def test_models_are_split_but_manual_entry_remains_a_ui_concern(self):
        chat, image = classify_models(["gpt-5.6", "gpt-image-2", "text-embedding-3-small", "flux-1"])
        self.assertEqual(chat, ["gpt-5.6"])
        self.assertEqual(image, ["flux-1", "gpt-image-2"])

    def test_fetch_uses_models_endpoint_and_redacts_key(self):
        client = Mock()
        client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="z"), SimpleNamespace(id="a")])
        self.assertEqual(fetch_model_ids("https://example.com/v1", "secret", client=client), ["a", "z"])
        client.models.list.side_effect = RuntimeError("bad secret")
        with self.assertRaises(ModelCatalogError) as context:
            fetch_model_ids("https://example.com/v1", "secret", client=client)
        self.assertNotIn("secret", str(context.exception))


if __name__ == "__main__":
    unittest.main()
