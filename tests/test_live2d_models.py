from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from live2d_models import scan_live2d_models


class Live2DModelScannerTests(unittest.TestCase):
    def test_discovers_hiyori_asset(self) -> None:
        models = scan_live2d_models(Path(__file__).parents[1] / "assets" / "live2d")
        hiyori = next(model for model in models if "Hiyori" in model.name)
        self.assertEqual(hiyori.name, "Hiyori / 日和")
        self.assertEqual(hiyori.generation, 3)
        self.assertEqual(hiyori.kind, "cubism")

    def test_discovers_wedding_hinata_avatar(self) -> None:
        models = scan_live2d_models(Path(__file__).parents[1] / "assets" / "live2d")
        hinata = next(model for model in models if "Wedding Hinata" in model.name)
        self.assertEqual(hinata.kind, "avatar")
        self.assertEqual(
            hinata.relative_manifest,
            "wedding_hinata_25d/wedding-hinata.avatar.json",
        )

    def test_rejects_missing_and_path_traversal_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid"
            valid.mkdir()
            (valid / "model.moc3").write_bytes(b"moc")
            (valid / "texture.png").write_bytes(b"png")
            (valid / "model.model3.json").write_text(
                json.dumps({"FileReferences": {"Moc": "model.moc3", "Textures": ["texture.png"]}}),
                encoding="utf-8",
            )
            unsafe = root / "unsafe"
            unsafe.mkdir()
            (unsafe / "model.model3.json").write_text(
                json.dumps({"FileReferences": {"Moc": "../outside.moc3", "Textures": ["texture.png"]}}),
                encoding="utf-8",
            )
            self.assertEqual(len(scan_live2d_models(root)), 1)
            self.assertEqual(scan_live2d_models(root)[0].relative_manifest, "valid/model.model3.json")


if __name__ == "__main__":
    unittest.main()
