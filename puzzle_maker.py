#!/usr/bin/env python3
"""
Tips & Tricks-style Pixel Puzzle Maker

Usage (examples):
  python puzzle_maker.py --input "https://example.com/image.png" --outdir output --sprite-size 16 --block 4 --grid 4 --scramble --rotations
  python puzzle_maker.py --input local_image.png --outdir output --sprite-size 32 --block 4 --grid 4

What it does:
1) Loads an image from a URL or local file.
2) Converts it to a pixelated sprite (e.g., 16x16 or 32x32), with optional palette reduction.
3) Slices the sprite into 4x4 mini-blocks.
4) Exports a scrambled puzzle sheet and a solution sheet.
"""

import argparse
import io
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as e:
    print("This script requires Pillow. Install with: pip install pillow")
    raise

try:
    import requests
except Exception:
    requests = None  # We'll gracefully handle if requests isn't available.


@dataclass
class BlockPlacement:
    index: int
    rotation: int  # degrees
    flip_h: bool
    flip_v: bool


def load_image(input_path: str) -> Image.Image:
    # URL or local file
    if input_path.lower().startswith(("http://", "https://")):
        if requests is None:
            raise RuntimeError("requests not installed; cannot fetch URLs. pip install requests")
        resp = requests.get(input_path, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        return img
    else:
        return Image.open(input_path).convert("RGBA")


def to_pixel_sprite(img: Image.Image, sprite_size: int, palette_colors: Optional[int]) -> Image.Image:
    # Center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Downscale to sprite_size x sprite_size using NEAREST, then upscale back for crisp pixels
    small = img.resize((sprite_size, sprite_size), resample=Image.NEAREST)

    if palette_colors and palette_colors > 1:
        # Quantize to shrink palette like old sprites
        small = small.convert("P", palette=Image.ADAPTIVE, colors=palette_colors).convert("RGBA")

    return small


def split_into_blocks(sprite: Image.Image, block_size: int) -> List[Image.Image]:
    W, H = sprite.size
    if W % block_size != 0 or H % block_size != 0:
        raise ValueError(f"Sprite size {W}x{H} must be divisible by block {block_size}.")
    blocks = []
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            blocks.append(sprite.crop((x, y, x + block_size, y + block_size)))
    return blocks


def make_canvas(cols: int, rows: int, cell_px: int, margin: int = 16, bg=(255,255,255,255)) -> Tuple[Image.Image, Tuple[int,int]]:
    W = margin*2 + cols*cell_px
    H = margin*2 + rows*cell_px
    canvas = Image.new("RGBA", (W, H), bg)
    return canvas, (margin, margin)


def draw_grid(canvas: Image.Image, top_left: Tuple[int,int], cols:int, rows:int, cell_px:int, line_color=(0,0,0,255)) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0 = top_left
    W = cols*cell_px
    H = rows*cell_px
    # outer
    draw.rectangle([x0, y0, x0+W, y0+H], outline=line_color, width=2)
    # inner
    for c in range(1, cols):
        x = x0 + c*cell_px
        draw.line([(x, y0), (x, y0+H)], fill=line_color, width=1)
    for r in range(1, rows):
        y = y0 + r*cell_px
        draw.line([(x0, y), (x0+W, y)], fill=line_color, width=1)


def place_block(img: Image.Image, block: Image.Image, cell_px: int, top_left: Tuple[int,int], col:int, row:int) -> None:
    x0, y0 = top_left
    x = x0 + col*cell_px
    y = y0 + row*cell_px
    # Scale block into cell
    block_resized = block.resize((cell_px, cell_px), resample=Image.NEAREST)
    img.alpha_composite(block_resized, (x, y))


def transform_block(block: Image.Image, rotation:int=0, flip_h:bool=False, flip_v:bool=False) -> Image.Image:
    b = block
    if flip_h:
        b = b.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        b = b.transpose(Image.FLIP_TOP_BOTTOM)
    if rotation % 360 != 0:
        b = b.rotate(rotation, expand=False, resample=Image.NEAREST)
    return b


def layout_puzzle(blocks: List[Image.Image], grid:int, cell_px:int, scramble:bool, allow_rotations:bool, seed:int=0) -> Tuple[Image.Image, Image.Image]:
    random.seed(seed)
    cols = rows = grid
    puzzle_canvas, origin = make_canvas(cols, rows, cell_px)
    sol_canvas, sol_origin = make_canvas(cols, rows, cell_px)

    # Draw grids
    draw_grid(puzzle_canvas, origin, cols, rows, cell_px)
    draw_grid(sol_canvas, sol_origin, cols, rows, cell_px)

    # Place solution blocks in order
    for i, block in enumerate(blocks):
        c = i % cols
        r = i // cols
        place_block(sol_canvas, block, cell_px, sol_origin, c, r)

    # Prepare placements
    indices = list(range(len(blocks)))
    if scramble:
        random.shuffle(indices)

    placements = []
    for idx in indices:
        rotation = 0
        flip_h = False
        flip_v = False
        if allow_rotations:
            rotation = random.choice([0, 90, 180, 270])
            flip_h = random.choice([False, True])
            flip_v = random.choice([False, True])
        placements.append(BlockPlacement(index=idx, rotation=rotation, flip_h=flip_h, flip_v=flip_v))

    # Place blocks into puzzle sheet
    for i, placement in enumerate(placements):
        c = i % cols
        r = i // cols
        blk = transform_block(blocks[placement.index], placement.rotation, placement.flip_h, placement.flip_v)
        place_block(puzzle_canvas, blk, cell_px, origin, c, r)

    return puzzle_canvas, sol_canvas


def save_blocks(blocks: List[Image.Image], outdir: Path, scale:int=16) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(blocks):
        b.resize((b.width*scale, b.height*scale), resample=Image.NEAREST).save(outdir / f"block_{i:02d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="URL or local path to source image")
    ap.add_argument("--outdir", default="output", help="Output directory")
    ap.add_argument("--sprite-size", type=int, default=16, help="Final sprite size in pixels (width=height). e.g., 16 or 32")
    ap.add_argument("--block", type=int, default=4, help="Mini-block dimension, e.g., 4 (must divide sprite-size)")
    ap.add_argument("--grid", type=int, default=4, help="Number of blocks across/down, e.g., 4 for 4x4 total layout")
    ap.add_argument("--cellpx", type=int, default=64, help="Pixel size of each block cell on sheets (visual scale)")
    ap.add_argument("--palette", type=int, default=16, help="Reduce colors (adaptive). Set 0 or 1 to skip.")
    ap.add_argument("--scramble", action="store_true", help="Scramble block order for the puzzle sheet")
    ap.add_argument("--rotations", action="store_true", help="Random rotations & flips in puzzle sheet (hard mode)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducible puzzles")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load
    img = load_image(args.input)

    # 2) sprite conversion
    sprite = to_pixel_sprite(img, sprite_size=args.sprite_size, palette_colors=args.palette)

    # 3) split
    if args.sprite_size % args.block != 0:
        print(f"--sprite-size ({args.sprite_size}) must be divisible by --block ({args.block}).", file=sys.stderr)
        sys.exit(2)

    blocks = split_into_blocks(sprite, args.block)

    # 4) save individual blocks (enlarged for clarity)
    save_blocks(blocks, outdir / "blocks", scale=max(1, args.cellpx // args.block))

    # 5) layouts
    puzzle_sheet, solution_sheet = layout_puzzle(
        blocks, grid=args.grid, cell_px=args.cellpx,
        scramble=args.scramble, allow_rotations=args.rotations, seed=args.seed
    )

    # 6) export sheets & sprite
    sprite.resize((args.sprite_size*8, args.sprite_size*8), Image.NEAREST).save(outdir / "final_sprite.png")
    puzzle_sheet.save(outdir / "puzzle_sheet.png")
    solution_sheet.save(outdir / "solution_sheet.png")

    # 7) write metadata
    meta = {
        "input": args.input,
        "sprite_size": args.sprite_size,
        "block": args.block,
        "grid": args.grid,
        "cellpx": args.cellpx,
        "palette": args.palette,
        "scramble": bool(args.scramble),
        "rotations": bool(args.rotations),
        "seed": args.seed
    }
    (outdir / "puzzle_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done! Outputs in: {outdir.resolve()}")
    print(" - final_sprite.png")
    print(" - puzzle_sheet.png")
    print(" - solution_sheet.png")
    print(" - blocks/*.png")
    print(" - puzzle_meta.json")


if __name__ == "__main__":
    main()
