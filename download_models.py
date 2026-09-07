"""Fetch only the runtime model files; verify them before installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def verified(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def prepare_models(root: Path, models: list[dict], *, check: bool = False, downloader=None) -> bool:
    all_ready = True
    for model in models:
        destination = root / "models" / model["directory"]
        for entry in model["files"]:
            target = destination / entry["path"]
            label = f'{model["directory"]}/{entry["path"]}'
            if verified(target, entry):
                print(f"[verified] {label}", flush=True)
                continue
            if check:
                print(f"[missing or invalid] {label}", flush=True)
                all_ready = False
                continue

            cache = root / ".cache" / "model-downloads"
            cache.mkdir(parents=True, exist_ok=True)
            if downloader is None:
                os.environ["MODELSCOPE_CACHE"] = str(root / ".cache" / "modelscope")
                from modelscope.hub.file_download import model_file_download

                downloader = model_file_download

            print(f'[download] {label} ({entry["bytes"] / 1_000_000:.1f} MB)', flush=True)
            # Stage on the same drive. A failed download cannot replace a model.
            with TemporaryDirectory(prefix="download-", dir=cache) as staging:
                downloaded = Path(downloader(
                    model_id=model["repo"],
                    file_path=entry["path"],
                    revision=model["revision"],
                    local_dir=staging,
                    cache_dir=str(cache),
                ))
                if not verified(downloaded, entry):
                    raise RuntimeError(
                        f"Checksum mismatch: {label}. Existing model was not replaced. "
                        "Retry or check the upstream revision against models-manifest.json."
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(downloaded, target)
            print(f"[installed] {label}", flush=True)
    return all_ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without downloading")
    parser.add_argument("--model", choices=["all", "paraformer-large", "FireRedVAD"], default="all")
    args = parser.parse_args()
    models = json.loads((ROOT / "models-manifest.json").read_text(encoding="utf-8"))["models"]
    if args.model != "all":
        models = [model for model in models if model["directory"] == args.model]
    try:
        ready = prepare_models(ROOT, models, check=args.check)
    except Exception as exc:
        print(f"Model setup failed: {exc}")
        return 1
    print("All requested models are ready." if ready else "Run download_models.py to download missing files.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
