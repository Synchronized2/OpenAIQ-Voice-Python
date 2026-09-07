from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from download_models import prepare_models


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"verified model data"
        self.entry = {
            "path": "Stream-VAD/model.pth.tar", "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.models = [{"directory": "test", "repo": "test/model", "revision": "v1", "files": [self.entry]}]
        self.target = self.root / "models" / "test" / self.entry["path"]

    def download(self, **kwargs):
        path = Path(kwargs["local_dir"]) / "downloaded"
        path.write_bytes(self.payload)
        return str(path)

    def test_download_and_skip_verified_file(self):
        downloader = Mock(side_effect=self.download)
        self.assertTrue(prepare_models(self.root, self.models, downloader=downloader))
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertTrue(prepare_models(self.root, self.models, downloader=downloader))
        downloader.assert_called_once()
        self.assertEqual(downloader.call_args.kwargs["revision"], "v1")

    def test_read_only_check_reports_missing_without_download(self):
        downloader = Mock()
        self.assertFalse(prepare_models(self.root, self.models, check=True, downloader=downloader))
        downloader.assert_not_called()
        self.assertFalse((self.root / ".cache").exists())

    def test_same_size_corruption_is_detected_and_repaired(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"X" * len(self.payload))
        self.assertFalse(prepare_models(self.root, self.models, check=True))
        prepare_models(self.root, self.models, downloader=self.download)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_bad_download_preserves_existing_file(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"existing data")
        self.entry["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
            prepare_models(self.root, self.models, downloader=self.download)
        self.assertEqual(self.target.read_bytes(), b"existing data")
        self.assertEqual(list((self.root / ".cache" / "model-downloads").iterdir()), [])

    def test_download_failure_can_be_retried(self):
        with self.assertRaises(OSError):
            prepare_models(self.root, self.models, downloader=Mock(side_effect=OSError("offline")))
        self.assertFalse(self.target.exists())
        self.assertTrue(prepare_models(self.root, self.models, downloader=self.download))


if __name__ == "__main__":
    unittest.main()
