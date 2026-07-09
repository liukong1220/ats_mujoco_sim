#!/usr/bin/env python3
"""Open the RMUC 2026 narrow tunnel gates to the real 0.78 m passable width."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_INPUT = Path("src/ats_sentry_bringup/map/rmuc_2026.pgm")
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_BACKUP = Path("src/ats_sentry_bringup/map/rmuc_2026_before_corridor_patch.pgm")
DEFAULT_PREVIEW = Path(
    "src/ats_sentry_bringup/map/obj_projection_preview/rmuc_2026_corridor_patch.png"
)
DEFAULT_RESOLUTION = 0.028466796875


@dataclass(frozen=True)
class Corridor:
    name: str
    box: tuple[int, int, int, int]


CORRIDORS = (
    # Pixel boxes are in the 1024x564 rmuc_2026.pgm image. For the long top
    # and bottom tunnels, keep both black boundary edges and only clear the
    # white passage between them.
    Corridor("top_center_tunnel", (461, 65, 550, 82)),
    Corridor("top_right_tunnel", (664, 87, 692, 105)),
    Corridor("lower_left_tunnel", (331, 459, 359, 479)),
    Corridor("bottom_center_tunnel", (486, 481, 584, 499)),
)


def patch_corridors(
    input_pgm: Path,
    output_pgm: Path,
    backup_pgm: Path | None,
    preview_png: Path | None,
    free_value: int,
) -> None:
    image = Image.open(input_pgm).convert("L")
    if image.size != (1024, 564):
        raise RuntimeError(f"Unexpected RMUC map size {image.size}; expected 1024x564")

    if backup_pgm is not None and input_pgm.resolve() == output_pgm.resolve():
        backup_pgm.parent.mkdir(parents=True, exist_ok=True)
        if not backup_pgm.exists():
            shutil.copy2(input_pgm, backup_pgm)

    patched = image.copy()
    draw = ImageDraw.Draw(patched)
    for corridor in CORRIDORS:
        draw.rectangle(corridor.box, fill=int(free_value))

    output_pgm.parent.mkdir(parents=True, exist_ok=True)
    patched.save(output_pgm)

    if preview_png is not None:
        preview = patched.convert("RGB")
        preview_draw = ImageDraw.Draw(preview)
        for corridor in CORRIDORS:
            preview_draw.rectangle(corridor.box, outline=(255, 0, 0), width=3)
            x0, y0, x1, y1 = corridor.box
            preview_draw.text((x0 + 3, max(0, y0 - 13)), corridor.name, fill=(255, 0, 0))
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        preview.save(preview_png)

    print(f"Input: {input_pgm}")
    print(f"Output: {output_pgm}")
    if backup_pgm is not None:
        print(f"Backup: {backup_pgm}")
    if preview_png is not None:
        print(f"Preview: {preview_png}")
    print(f"Free value: {free_value}")
    for corridor in CORRIDORS:
        x0, y0, x1, y1 = corridor.box
        x_size_m = (x1 - x0) * DEFAULT_RESOLUTION
        y_size_m = (y1 - y0) * DEFAULT_RESOLUTION
        print(
            f"{corridor.name}: box={corridor.box}, "
            f"x_size={x_size_m:.3f} m, y_size={y_size_m:.3f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--free-value", type=int, default=254)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    patch_corridors(
        input_pgm=args.input,
        output_pgm=args.output,
        backup_pgm=None if args.no_backup else args.backup,
        preview_png=None if args.no_preview else args.preview,
        free_value=args.free_value,
    )


if __name__ == "__main__":
    main()
