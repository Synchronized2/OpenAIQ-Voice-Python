from __future__ import annotations

import unittest

from agent import LocalAction, is_voice_exit_phrase, parse_local_intent


class LocalIntentTests(unittest.TestCase):
    def assert_action(self, text: str, kind: str, value=None) -> None:  # noqa: ANN001
        self.assertEqual(parse_local_intent(text), LocalAction(kind, value))

    def test_exact_volume_commands(self) -> None:
        self.assert_action("set volume to 33", "set_volume", 33)
        self.assert_action("把音量调到 45%", "set_volume", 45)
        self.assert_action("帮我把音量提高到60%", "set_volume", 60)
        self.assert_action("请把系统音量降低到百分之二十五", "set_volume", 25)
        self.assert_action("音量调到三十三", "set_volume", 33)
        self.assert_action("音量调高一点", "change_volume", 10)
        self.assert_action("帮我把音量提高一点", "change_volume", 10)
        self.assert_action("帮我把音量提高10%", "change_volume", 10)
        self.assert_action("降低音量 20%", "change_volume", -20)
        self.assert_action("静音", "mute", True)
        self.assert_action("取消静音", "mute", False)

    def test_brightness_and_media_commands(self) -> None:
        self.assert_action("亮度设置为 60", "set_brightness", 60)
        self.assert_action("屏幕亮度调到百分之七十五", "set_brightness", 75)
        self.assert_action("帮我把亮度提高到80%", "set_brightness", 80)
        self.assert_action("帮我把亮度提高一点", "change_brightness", 10)
        self.assert_action("brightness down", "change_brightness", -10)
        self.assert_action("下一首", "media", "next")
        self.assert_action("暂停播放", "media", "pause")
        self.assert_action("显示桌面", "show_desktop")

    def test_application_name_is_not_truncated(self) -> None:
        self.assert_action("open qq music", "open_application", "qq music")
        self.assert_action("打开QQ音乐", "open_application", "QQ音乐")
        self.assertNotEqual(
            parse_local_intent("open qq music"),
            LocalAction("open_application", "qq"),
        )

    def test_conversation_is_not_treated_as_an_action(self) -> None:
        for text in (
            "为什么相机录音的音量这么小",
            "怎么保护眼睛和调节亮度比较好",
            "你觉得 QQ 音乐怎么样",
            "我刚才打开了记事本",
            "请解释 set volume to 33 是什么意思",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_local_intent(text))

    def test_voice_exit_requires_a_complete_phrase(self) -> None:
        for text in ("拜拜", "再见！", "先这样吧", "stop listening"):
            self.assertTrue(is_voice_exit_phrase(text))
        self.assertFalse(is_voice_exit_phrase("拜拜是什么意思"))
        self.assertFalse(is_voice_exit_phrase("给朋友说再见"))


if __name__ == "__main__":
    unittest.main()
