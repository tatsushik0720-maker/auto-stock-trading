"""対戦画面のアプリアイコンを描く（src/sim/assets/ に PNG を作る）。

絵: 紺の背景に、青（戦略くん）とオレンジ（放置くん）の2本の線が
    競り合いながら右上へ伸び、先頭に点。買いの ▲ を1つ。
色は画面と同じ M3 の配色（暗い表示の primary #9ECAFF と放置くんの #FFB870）。

使い方:
    python scripts/make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "src" / "sim" / "assets"
S = 4096                       # 大きく描いてから縮める（線のふちをなめらかにする）
K = S / 1024

BG_TOP, BG_BOTTOM = (0, 73, 125), (0, 29, 54)        # primary-container(dark) → on-primary-container
GRID = (209, 228, 255, 34)
STRAT = (158, 202, 255)       # 戦略くん（primary, dark）
BENCH = (255, 184, 112)       # 放置くん
BUY = (120, 220, 119)


def draw_icon(scale: float = 1.0) -> Image.Image:
    """scale < 1 で絵を中央に縮めて描く（maskable 用の余白。画像を縮めずに座標を縮める）。"""
    img = Image.new("RGB", (S, S))
    grad = ImageDraw.Draw(img)
    for y in range(S):
        t = y / (S - 1)
        grad.line([(0, y), (S, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    bg_mid = tuple(round((a + b) / 2) for a, b in zip(BG_TOP, BG_BOTTOM))

    def xy(x, y):
        return ((512 + (x - 512) * scale) * K, (512 + (y - 512) * scale) * K)

    def w(v):
        return int(v * scale * K)

    art = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    for y in (330, 520, 710):                          # うすい横線
        d.line([xy(170, y), xy(854, y)], fill=GRID, width=w(10))

    bench = [(170, 700), (320, 630), (450, 670), (600, 520), (730, 480), (840, 430)]
    strat = [(170, 770), (310, 590), (440, 560), (580, 640), (710, 450), (840, 290)]
    for pts, color in ((bench, BENCH), (strat, STRAT)):
        d.line([xy(*q) for q in pts], fill=color, width=w(58), joint="curve")
        cx, cy = xy(*pts[0])                           # 線の始まりを丸くする
        r = w(29)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    for pts, color in ((bench, BENCH), (strat, STRAT)):   # 先頭の点（背景の色で縁取り）
        cx, cy = xy(*pts[-1])
        r1, r2 = w(70), w(50)
        d.ellipse((cx - r1, cy - r1, cx + r1, cy + r1), fill=bg_mid)
        d.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), fill=color)
    bx, by, h = 580, 735, 60                           # 買いの ▲（戦略くんの線の下）
    d.polygon([xy(bx, by - h), xy(bx - h * 1.05, by + h * .8), xy(bx + h * 1.05, by + h * .8)], fill=BUY)

    img.paste(art, (0, 0), art)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = draw_icon()
    for n in (180, 192, 512):
        full.resize((n, n), Image.LANCZOS).save(OUT / f"icon-{n}.png", optimize=True)
    # Android で丸や角丸に切り抜かれても欠けないよう、絵を 80% に縮めたもの
    draw_icon(0.8).resize((512, 512), Image.LANCZOS).save(OUT / "icon-maskable-512.png", optimize=True)
    print("作成:", ", ".join(sorted(f.name for f in OUT.glob("icon-*.png"))))


if __name__ == "__main__":
    main()
