from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "new_models" / "png"
OUTPUT = ROOT / "assets" / "live2d" / "wedding_hinata_25d"
PREVIEW = ROOT / ".cache" / "live2d-build" / "wedding-hinata-preview.png"

# Coordinates in the supplied expression reference sheet.
EXPRESSION_CROPS = {
    "smile": (456, 500, 558, 623),
    "happy": (563, 500, 665, 623),
    "angry": (670, 500, 773, 623),
    "sad": (348, 648, 451, 787),
    "surprised": (456, 648, 558, 787),
    "blink": (563, 648, 665, 787),
    "shocked": (670, 648, 773, 787),
}
FACE_DESTINATION = (350, 34, 674, 446)


def source_images() -> tuple[Path, Path]:
    images = list(SOURCE.glob("*.png"))
    by_size: dict[tuple[int, int], Path] = {}
    for path in images:
        with Image.open(path) as image:
            by_size[image.size] = path
    try:
        return by_size[(1024, 1536)], by_size[(1214, 1295)]
    except KeyError as exc:
        raise SystemExit("Expected the supplied 1024x1536 portrait and 1214x1295 reference sheet") from exc


def feathered_overlay(reference: Image.Image, crop: tuple[int, int, int, int]) -> Image.Image:
    width = FACE_DESTINATION[2] - FACE_DESTINATION[0]
    height = FACE_DESTINATION[3] - FACE_DESTINATION[1]
    face = reference.crop(crop).resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", (width, height))
    drawing = ImageDraw.Draw(mask)
    drawing.rounded_rectangle((18, 10, width - 18, height - 12), radius=90, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(18))
    face.putalpha(mask)
    overlay = Image.new("RGBA", (1024, 1536))
    overlay.alpha_composite(face, FACE_DESTINATION[:2])
    return overlay


def main() -> None:
    portrait_path, reference_path = source_images()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(portrait_path) as portrait_source, Image.open(reference_path) as reference_source:
        portrait = portrait_source.convert("RGB")
        reference = reference_source.convert("RGB")
        portrait.save(OUTPUT / "portrait.webp", "WEBP", quality=95, method=6)
        preview_tiles = []
        for name, crop in EXPRESSION_CROPS.items():
            overlay = feathered_overlay(reference, crop)
            overlay.save(OUTPUT / f"expression-{name}.png", optimize=True)
            preview = portrait.convert("RGBA")
            preview.alpha_composite(overlay)
            preview.thumbnail((256, 384), Image.Resampling.LANCZOS)
            preview_tiles.append((name, preview.copy()))

    manifest = {
        "format": "openaiq-2.5d-avatar",
        "version": 1,
        "name": "婚纱雏田 / Wedding Hinata",
        "image": "portrait.webp",
        "expressions": {
            "listening": "expression-surprised.png",
            "thinking": "expression-smile.png",
            "executing": "expression-angry.png",
            "speaking": "expression-happy.png",
            "error": "expression-sad.png",
            "blink": "expression-blink.png",
            "shocked": "expression-shocked.png",
        },
        "face": {"mouth": [512, 326], "blush": [512, 302]},
        "view": {"zoom": 1.18, "offsetY": 0.08, "wideZoom": 1.55, "wideOffsetY": 0.3},
        "motion": {"breath": 0.006, "sway": 0.012, "pointer": 0.018},
    }
    (OUTPUT / "wedding-hinata.avatar.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sheet = Image.new("RGB", (256 * 4, 410 * 2), "#eef0f6")
    drawing = ImageDraw.Draw(sheet)
    for index, (name, tile) in enumerate(preview_tiles):
        x = index % 4 * 256
        y = index // 4 * 410
        sheet.paste(tile.convert("RGB"), (x, y))
        drawing.text((x + 10, y + 384), name, fill="#24283a")
    sheet.save(PREVIEW, optimize=True)
    print(f"Built {OUTPUT}")
    print(f"Preview {PREVIEW}")


if __name__ == "__main__":
    main()
