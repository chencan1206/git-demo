"""Generate application icons (PNG + ICO) with no external assets.

Run:  python packaging/make_icon.py
Out:  packaging/icon_512.png, packaging/icon.ico, packaging/icon_256.png
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent
SIZE = 512


def gradient(size: int) -> Image.Image:
    """Diagonal green -> orange gradient."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    assert px is not None
    c1 = (39, 174, 96)     # green
    c2 = (230, 126, 34)    # orange
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = (
                round(c1[0] + (c2[0] - c1[0]) * t),
                round(c1[1] + (c2[1] - c1[1]) * t),
                round(c1[2] + (c2[2] - c1[2]) * t),
            )
    return img


def leaf_mask(size: int) -> Image.Image:
    """Lens-shaped leaf with a midrib, as an alpha mask."""
    m = Image.new("L", (size * 4, size * 4), 0)
    d = ImageDraw.Draw(m)
    S = size * 4
    # leaf outline: two mirrored quadratics approximated with polygons
    top, bottom = S * 0.12, S * 0.88
    left, right = S * 0.20, S * 0.80
    pts_up = []
    pts_dn = []
    for i in range(81):
        t = i / 80
        x = left + (right - left) * t
        bulge = math.sin(math.pi * t) * (S * 0.24)
        pts_up.append((x, S * 0.5 - bulge))
        pts_dn.append((x, S * 0.5 + bulge * 0.82))
    d.polygon(pts_up + pts_dn[::-1], fill=255)
    # midrib + veins punched out
    d.line([(left, S * 0.5), (right, S * 0.5)], fill=0, width=int(S * 0.018))
    for k in range(1, 6):
        t = k / 6.5
        x = left + (right - left) * t
        bulge = math.sin(math.pi * t) * (S * 0.24)
        d.line([(x, S * 0.5 - bulge * 0.85), (x + S * 0.05, S * 0.5 - S * 0.03)],
               fill=0, width=int(S * 0.009))
        d.line([(x, S * 0.5 + bulge * 0.70), (x + S * 0.05, S * 0.5 + S * 0.03)],
               fill=0, width=int(S * 0.009))
    _ = (top, bottom)
    return m.resize((size, size), Image.LANCZOS)


def rounded_alpha(size: int, radius: int) -> Image.Image:
    a = Image.new("L", (size, size), 0)
    ImageDraw.Draw(a).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return a


def main() -> int:
    base = gradient(SIZE)
    base = base.filter(ImageFilter.GaussianBlur(0.4))

    icon = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    icon.paste(base, (0, 0), rounded_alpha(SIZE, 96))

    leaf = leaf_mask(SIZE)
    white = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 235))
    icon = Image.alpha_composite(icon, Image.composite(
        white, Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0)), leaf))

    icon.save(OUT / "icon_512.png")
    icon.resize((256, 256), Image.LANCZOS).save(OUT / "icon_256.png")
    icon.save(OUT / "icon.ico",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                     (128, 128), (256, 256)])
    print("icons written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
