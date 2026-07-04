"""
Synthetic airport signage generator.

Produces mockup images for every record in the airport knowledge base, so
the visual dataset maps 1:1 onto the KB used by the text pipeline. This is
one of the data sources explicitly permitted by the brief: "self-curated
images, synthetic images, or screenshots from airport style mockups."

Why synthetic mockups rather than only scraped photos:
  - The airport (Vion International) is fictional, so no real photo of
    "Gate B12" or "Vion International" signage exists.
  - Guarantees a clean, license-free, exactly-labelled dataset: every image
    is tied to a real KB record id from the start, so no manual re-labelling
    or licence tracking is needed.
  - Real open-licence photos can still be added later (see README note at
    the bottom of this file) to discuss the synthetic-vs-real domain gap in
    the report's "dataset limitations" section - that comparison is only
    possible once a synthetic baseline exists.

For each record, FOUR images are produced:
  <id>_v1.png                 - a clean reference image (the CLIP "gallery"
                                 image, compared against the KB's
                                 visual_description at inference time).
  <id>_v2_angle.png           - a query image simulating a hurried, off-angle
                                 phone photo (larger rotation, mild blur).
  <id>_v3_dark.png            - a query image simulating a low-light terminal
                                 photo (strong brightness reduction).
  <id>_v4_blur.png            - a query image simulating a motion-blurred
                                 photo, e.g. taken while walking (heavy blur).

Having three distinct, labelled degradation conditions (rather than one
generic "degraded" variant) means evaluation can report accuracy broken down
by condition - mirroring how the text pipeline broke queries down by
easy/hard difficulty - instead of a single averaged robustness number.

Images are saved under data/images/<category>/<id>_v{1,2}.png, using the
KB's own category field as the folder name - this matches the annotation
categories the brief itself suggests (gate, baggage_claim, check_in,
security, restroom, lounge, transport, information_desk, restaurant, ...).

Run:  python -m data.images.generate_mockups
(from the project root; requires Pillow only, no network access)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)  # reproducible degraded variants

ROOT_DIR = Path(__file__).resolve().parents[2]
KB_PATH = ROOT_DIR / "data" / "knowledge_base" / "airport_kb.json"
OUT_DIR = ROOT_DIR / "data" / "images"

CANVAS_SIZE = (1000, 300)


def _find_font(candidates: list[str]) -> str:
    """Return the first existing font path from a cross-platform candidate list."""
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(
        "No usable font found. Place a .ttf font (e.g. DejaVuSans-Bold.ttf) "
        "next to this script and add its path to FONT_BOLD/FONT_REGULAR below, "
        "or install one of: Arial (Windows), DejaVu Sans (Linux)."
    )


FONT_BOLD = _find_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux
    "C:/Windows/Fonts/arialbd.ttf",                            # Windows (Arial Bold)
    "C:/Windows/Fonts/segoeuib.ttf",                           # Windows (Segoe UI Bold)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",       # macOS
])
FONT_REGULAR = _find_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Linux
    "C:/Windows/Fonts/arial.ttf",                              # Windows
    "C:/Windows/Fonts/segoeui.ttf",                            # Windows (Segoe UI)
    "/System/Library/Fonts/Supplemental/Arial.ttf",            # macOS
])

# Layout constants for the wayfinding-sign style (rounded icon box on the
# left, bold title + underline + subtitle in the middle, direction arrow on
# the right), matching a real airport gate/baggage sign layout.
OUTER_MARGIN = 10
OUTER_RADIUS = 26
INNER_RADIUS = 20
ICON_BOX = (40, 55, 220, 235)          # x0, y0, x1, y1
ARROW_BOX = (860, 105, 950, 195)       # x0, y0, x1, y1
TEXT_X_START = 260
TEXT_X_END = 840


def _lighten(hex_color: str, factor: float = 0.35) -> tuple[int, int, int]:
    """Blend a hex colour with white; used for the sign's outer border ring."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (int(r + (255 - r) * factor), int(g + (255 - g) * factor), int(b + (255 - b) * factor))


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int,
              start_size: int, min_size: int = 26) -> ImageFont.FreeTypeFont:
    """Shrink the bold font until `text` fits max_width (fixes overflow on long names)."""
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(FONT_BOLD, size)
        w = draw.textbbox((0, 0), text, font=font)[2]
        if w <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(FONT_BOLD, min_size)


def _draw_arrow(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str) -> None:
    """Solid right-pointing wayfinding arrow, filled, matching sign-arrow style."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    shaft_h = h * 0.28
    cy = (y0 + y1) / 2
    shaft_right = x0 + (x1 - x0) * 0.55
    draw.rectangle([x0, cy - shaft_h / 2, shaft_right, cy + shaft_h / 2], fill=color)
    draw.polygon([(shaft_right - 2, y0), (x1, cy), (shaft_right - 2, y1)], fill=color)

# One background colour per category, loosely following real airport
# wayfinding conventions (blue = general information/direction,
# green = currency/money, warm tones = food, muted = premium/lounge).
CATEGORY_STYLE = {
    "gate":               "#0B3D91",
    "check_in":           "#0B3D91",
    "baggage_claim":      "#0B3D91",
    "security":           "#1B1F3B",
    "information_desk":   "#0B3D91",
    "lounge":             "#5B2A6E",
    "restaurant":         "#A8511B",
    "restroom":           "#0B3D91",
    "prayer_room":        "#2E6E62",
    "lost_and_found":     "#4A4A4A",
    "customs":            "#0B3D91",
    "currency_exchange":  "#1E7A3C",
    "special_assistance": "#0B3D91",
    "transport":          "#0B3D91",
}
TEXT_COLOR = "#FFFFFF"
DEFAULT_BG = "#0B3D91"


def _icon(draw: ImageDraw.ImageDraw, category: str, cx: int, cy: int, s: int) -> None:
    """
    Draw a simple geometric pictogram for the category, centred at (cx, cy)
    with characteristic size s. These are intentionally simplified shapes
    (not exact ISO/AIGA icon reproductions) built from basic primitives, to
    avoid any icon-set licensing questions while still giving CLIP a
    consistent visual cue per category.
    """
    c = TEXT_COLOR
    if category == "gate":
        # simple airplane silhouette: a triangle body + two wing triangles
        draw.polygon([(cx - s, cy), (cx + s, cy - s * 0.15), (cx + s, cy + s * 0.15)], fill=c)
        draw.polygon([(cx, cy - s * 0.1), (cx - s * 0.3, cy - s * 0.7), (cx + s * 0.1, cy - s * 0.1)], fill=c)
        draw.polygon([(cx, cy + s * 0.1), (cx - s * 0.3, cy + s * 0.7), (cx + s * 0.1, cy + s * 0.1)], fill=c)
    elif category == "check_in":
        draw.rectangle([cx - s, cy - s * 0.6, cx + s, cy + s * 0.6], outline=c, width=6)
        draw.line([(cx - s, cy), (cx + s, cy)], fill=c, width=4)
    elif category == "baggage_claim":
        draw.rounded_rectangle([cx - s * 0.7, cy - s * 0.5, cx + s * 0.7, cy + s * 0.7], radius=12, outline=c, width=6)
        draw.rectangle([cx - s * 0.25, cy - s * 0.8, cx + s * 0.25, cy - s * 0.5], outline=c, width=5)
    elif category == "security":
        draw.polygon([(cx, cy - s), (cx + s * 0.8, cy - s * 0.4), (cx + s * 0.8, cy + s * 0.4),
                       (cx, cy + s), (cx - s * 0.8, cy + s * 0.4), (cx - s * 0.8, cy - s * 0.4)], outline=c, width=6)
        draw.line([(cx - s * 0.3, cy), (cx - s * 0.05, cy + s * 0.3), (cx + s * 0.4, cy - s * 0.3)], fill=c, width=6)
    elif category == "information_desk":
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], outline=c, width=6)
        f = ImageFont.truetype(FONT_BOLD, int(s * 1.3))
        draw.text((cx, cy), "i", font=f, fill=c, anchor="mm")
    elif category == "lounge":
        draw.rounded_rectangle([cx - s, cy - s * 0.3, cx + s, cy + s * 0.6], radius=14, outline=c, width=6)
        draw.rounded_rectangle([cx - s, cy - s * 0.9, cx - s * 0.4, cy - s * 0.2], radius=10, outline=c, width=6)
        draw.rounded_rectangle([cx + s * 0.4, cy - s * 0.9, cx + s, cy - s * 0.2], radius=10, outline=c, width=6)
    elif category == "restaurant":
        draw.line([(cx - s * 0.5, cy - s), (cx - s * 0.5, cy + s)], fill=c, width=5)
        for dx in (-s * 0.65, -s * 0.5, -s * 0.35):
            draw.line([(cx + dx, cy - s), (cx + dx, cy - s * 0.3)], fill=c, width=4)
        draw.ellipse([cx + s * 0.2, cy - s, cx + s * 0.7, cy - s * 0.3], outline=c, width=5)
        draw.line([(cx + s * 0.45, cy - s * 0.3), (cx + s * 0.45, cy + s)], fill=c, width=5)
    elif category == "restroom":
        draw.ellipse([cx - s * 0.9, cy - s, cx - s * 0.5, cy - s * 0.6], fill=c)
        draw.polygon([(cx - s * 0.9, cy - s * 0.55), (cx - s * 0.5, cy - s * 0.55), (cx - s * 0.7, cy + s * 0.6)], fill=c)
        draw.ellipse([cx + s * 0.5, cy - s, cx + s * 0.9, cy - s * 0.6], fill=c)
        draw.polygon([(cx + s * 0.4, cy + s * 0.6), (cx + s * 1.0, cy + s * 0.6), (cx + s * 0.85, cy - s * 0.5),
                       (cx + s * 0.55, cy - s * 0.5)], fill=c)
    elif category == "prayer_room":
        draw.arc([cx - s, cy - s, cx + s, cy + s * 0.4], start=180, end=360, fill=c, width=8)
        draw.line([(cx - s, cy + s * 0.4), (cx - s, cy + s)], fill=c, width=6)
        draw.line([(cx + s, cy + s * 0.4), (cx + s, cy + s)], fill=c, width=6)
    elif category == "lost_and_found":
        f = ImageFont.truetype(FONT_BOLD, int(s * 1.6))
        draw.text((cx, cy), "?", font=f, fill=c, anchor="mm")
    elif category == "customs":
        draw.rounded_rectangle([cx - s * 0.6, cy - s, cx + s * 0.6, cy + s], radius=8, outline=c, width=6)
        for i, dy in enumerate((-s * 0.4, -s * 0.05, s * 0.3)):
            draw.line([(cx - s * 0.35, cy + dy), (cx + s * 0.35, cy + dy)], fill=c, width=4)
    elif category == "currency_exchange":
        f = ImageFont.truetype(FONT_BOLD, int(s * 1.6))
        draw.text((cx, cy), "$", font=f, fill=c, anchor="mm")
    elif category == "special_assistance":
        draw.ellipse([cx - s, cy - s * 0.9, cx - s * 0.5, cy - s * 0.4], outline=c, width=6)
        draw.arc([cx - s * 0.6, cy - s * 0.2, cx + s * 0.9, cy + s], start=0, end=180, fill=c, width=8)
        draw.line([(cx + s * 0.9, cy + s * 0.6), (cx + s * 0.9, cy - s * 0.1)], fill=c, width=6)
    elif category == "transport":
        draw.rounded_rectangle([cx - s, cy - s * 0.4, cx + s, cy + s * 0.5], radius=10, outline=c, width=6)
        draw.ellipse([cx - s * 0.7, cy + s * 0.4, cx - s * 0.3, cy + s * 0.8], outline=c, width=5)
        draw.ellipse([cx + s * 0.3, cy + s * 0.4, cx + s * 0.7, cy + s * 0.8], outline=c, width=5)
    else:
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=c, width=6)


def make_sign(record: dict) -> Image.Image:
    """
    Render the clean reference sign (v1) for a single KB record, styled as a
    real airport wayfinding sign: rounded outer border, a rounded icon box
    on the left, a bold auto-fitted title with a thin underline, a smaller
    subtitle line, and a directional arrow on the right.
    """
    bg = CATEGORY_STYLE.get(record["category"], DEFAULT_BG)
    border_color = _lighten(bg, factor=0.30)

    # White canvas so the rounded sign sits on a clean background (like a
    # sign photographed/cropped against a wall), then the sign itself.
    img = Image.new("RGB", CANVAS_SIZE, "#FFFFFF")
    draw = ImageDraw.Draw(img)

    outer = (OUTER_MARGIN, OUTER_MARGIN, CANVAS_SIZE[0] - OUTER_MARGIN, CANVAS_SIZE[1] - OUTER_MARGIN)
    draw.rounded_rectangle(outer, radius=OUTER_RADIUS, fill=border_color)
    inset = 9
    inner = (outer[0] + inset, outer[1] + inset, outer[2] - inset, outer[3] - inset)
    draw.rounded_rectangle(inner, radius=INNER_RADIUS, fill=bg)

    # Icon box (rounded square outline) with the category pictogram inside.
    draw.rounded_rectangle(ICON_BOX, radius=18, outline=TEXT_COLOR, width=4)
    icx = (ICON_BOX[0] + ICON_BOX[2]) / 2
    icy = (ICON_BOX[1] + ICON_BOX[3]) / 2
    icon_size = (ICON_BOX[2] - ICON_BOX[0]) * 0.30
    _icon(draw, record["category"], icx, icy, icon_size)

    # Title, auto-shrunk to fit between the icon box and the arrow box.
    max_text_width = TEXT_X_END - TEXT_X_START
    title_font = _fit_font(draw, record["name"], max_text_width, start_size=64, min_size=30)
    title_y = 105
    draw.text((TEXT_X_START, title_y), record["name"], font=title_font, fill=TEXT_COLOR, anchor="lm")

    # Thin underline beneath the title, then the subtitle (terminal) below it.
    underline_y = 155
    draw.line([(TEXT_X_START, underline_y), (TEXT_X_END, underline_y)], fill=TEXT_COLOR, width=3)

    sub_font = ImageFont.truetype(FONT_REGULAR, 34)
    draw.text((TEXT_X_START, 190), record["terminal"], font=sub_font, fill=TEXT_COLOR, anchor="lm")

    _draw_arrow(draw, ARROW_BOX, TEXT_COLOR)
    return img


def degrade(img: Image.Image, condition: str) -> Image.Image:
    """
    Produce a degraded 'passenger photo' variant under one of three distinct,
    labelled real-world conditions (mirrors the text pipeline's easy/hard
    query tagging, so the same kind of breakdown analysis is possible here):

      angle - a hurried, off-angle phone photo (larger rotation, mild blur).
      dark  - a low-light terminal photo (strong brightness reduction).
      blur  - a motion-blurred photo, e.g. taken while walking (heavy blur).

    Each condition is deliberately distinct so evaluation can later report
    accuracy broken down by condition, not just an averaged "degraded" score.
    """
    if condition == "angle":
        angle, blur_radius, brightness = random.uniform(-12, 12), random.uniform(0.3, 0.8), random.uniform(0.9, 1.05)
    elif condition == "dark":
        angle, blur_radius, brightness = random.uniform(-3, 3), random.uniform(0.2, 0.6), random.uniform(0.45, 0.65)
    elif condition == "blur":
        angle, blur_radius, brightness = random.uniform(-3, 3), random.uniform(1.8, 2.6), random.uniform(0.85, 1.05)
    else:
        raise ValueError(f"unknown degradation condition: {condition}")

    out = img.rotate(angle, expand=False, fillcolor=img.getpixel((0, 0)))
    out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = Image.eval(out, lambda p: max(0, min(255, int(p * brightness))))
    return out


DEGRADE_CONDITIONS = ["angle", "dark", "blur"]


def main() -> None:
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    manifest = []

    for record in kb["records"]:
        category_dir = OUT_DIR / record["category"]
        category_dir.mkdir(parents=True, exist_ok=True)

        v1 = make_sign(record)
        v1_path = category_dir / f"{record['id']}_v1.png"
        v1.save(v1_path)
        manifest.append({
            "filename": str(v1_path.relative_to(ROOT_DIR)),
            "record_id": record["id"],
            "category": record["category"],
            "variant": "reference",
            "condition": "clean",
        })

        for i, condition in enumerate(DEGRADE_CONDITIONS, start=2):
            variant_img = degrade(v1, condition)
            variant_path = category_dir / f"{record['id']}_v{i}_{condition}.png"
            variant_img.save(variant_path)
            manifest.append({
                "filename": str(variant_path.relative_to(ROOT_DIR)),
                "record_id": record["id"],
                "category": record["category"],
                "variant": "query",
                "condition": condition,
            })

    manifest_path = OUT_DIR / "image_labels.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("filename,record_id,category,variant,condition\n")
        for row in manifest:
            f.write(f"{row['filename']},{row['record_id']},{row['category']},{row['variant']},{row['condition']}\n")

    n_records = len(kb["records"])
    print(f"Generated {len(manifest)} images for {n_records} records "
          f"({n_records} reference + {n_records * len(DEGRADE_CONDITIONS)} query images "
          f"across conditions: {', '.join(DEGRADE_CONDITIONS)}).")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()