from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Live2DModel:
    """A validated local Live2D model manifest."""

    id: str
    name: str
    manifest: Path
    root: Path
    generation: int
    kind: str = "cubism"

    @property
    def relative_manifest(self) -> str:
        return self.manifest.relative_to(self.root).as_posix()


def default_model_root(base: Path | None = None) -> Path:
    return (base or Path(__file__).resolve().parent) / "assets" / "live2d"


def _descriptor(data: Any) -> tuple[int, str, list[str], str, str | None] | None:
    if not isinstance(data, dict):
        return None
    if data.get("format") == "openaiq-2.5d-avatar" and isinstance(data.get("image"), str):
        expressions = data.get("expressions", {})
        if not isinstance(expressions, dict) or any(not isinstance(value, str) for value in expressions.values()):
            return None
        name = data.get("name") if isinstance(data.get("name"), str) else None
        return 0, data["image"], list(expressions.values()), "avatar", name
    modern = data.get("FileReferences")
    if isinstance(modern, dict) and isinstance(modern.get("Moc"), str) and isinstance(modern.get("Textures"), list):
        return 3, modern["Moc"], modern["Textures"], "cubism", None
    if isinstance(data.get("model"), str) and isinstance(data.get("textures"), list):
        return 2, data["model"], data["textures"], "cubism", None
    return None


def _resolve_resource(root: Path, manifest: Path, reference: str) -> Path:
    if not reference or "\x00" in reference or "://" in reference:
        raise ValueError("invalid resource reference")
    path = Path(reference.replace("\\", "/"))
    if path.is_absolute() or (len(path.parts) > 1 and path.parts[0].endswith(":")):
        raise ValueError("absolute resource reference")
    resolved = (manifest.parent / path).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("resource leaves model directory")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(reference)
    return resolved


def _model_id(root: Path, manifest: Path) -> str:
    digest = hashlib.sha256(manifest.relative_to(root).as_posix().encode("utf-8")).hexdigest()[:12]
    return f"live2d-{digest}"


def scan_live2d_models(root: str | Path) -> list[Live2DModel]:
    """Recursively find valid Cubism and OpenAIQ 2.5D manifests below *root*.

    Only model and texture references are required for selection. Optional
    motion, expression and physics files are left to the browser runtime so a
    partially packaged model can still be previewed when its core assets work.
    """

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    models: list[Live2DModel] = []
    candidates = sorted(
        set(base.rglob("*.model.json"))
        | set(base.rglob("*.model3.json"))
        | set(base.rglob("*.avatar.json"))
        | set(base.rglob("index.json")),
        key=lambda path: path.as_posix().lower(),
    )
    for manifest in candidates:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            descriptor = _descriptor(data)
            if descriptor is None:
                continue
            generation, core, textures, kind, display_name = descriptor
            _resolve_resource(base, manifest, core)
            if not textures or any(not isinstance(item, str) for item in textures):
                continue
            for texture in textures:
                _resolve_resource(base, manifest, texture)
        except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError):
            continue
        relative = manifest.relative_to(base)
        name = display_name or relative.parent.name or manifest.stem
        if name.lower() in {"runtime", "model", "models"}:
            name = manifest.stem.replace(".model3", "").replace(".model", "")
        if "hiyori" in relative.as_posix().lower():
            name = "Hiyori / 日和"
        models.append(Live2DModel(_model_id(base, manifest), name, manifest, base, generation, kind))
    return models


def model_by_id(models: list[Live2DModel], model_id: str | None) -> Live2DModel | None:
    return next((model for model in models if model.id == model_id), None)
