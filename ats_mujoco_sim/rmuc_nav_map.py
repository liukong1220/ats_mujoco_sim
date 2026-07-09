#!/usr/bin/env python3
"""Refine the RMUC 2026 Nav2 occupancy grid against the MuJoCo hfield."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


DEFAULT_INPUT = Path("src/ats_sentry_bringup/map/rmuc_2026.pgm")
DEFAULT_HFIELD = Path("src/sim/ats_mujoco_sim/models/rmuc_2026_height.png")
DEFAULT_OUTPUT = Path(
    "src/ats_sentry_bringup/map/obj_projection_preview/rmuc_2026_refined_preview.pgm"
)
DEFAULT_BACKUP = Path("src/ats_sentry_bringup/map/rmuc_2026_before_obj_refine.pgm")
DEFAULT_ORIENTATION = "normal"
ORIENTATIONS = (
    "normal",
    "flip_x_left_right",
    "flip_y_up_down",
    "flip_xy_rotate_180",
)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def refine_map(
    input_pgm: Path,
    hfield_png: Path,
    output_pgm: Path,
    backup_pgm: Path | None,
    hfield_threshold: int,
    min_obj_area: int,
    orientation: str,
) -> None:
    manual = np.array(Image.open(input_pgm).convert("L"))
    hfield = np.array(Image.open(hfield_png).convert("L"))
    if orientation == "flip_x_left_right":
        hfield = np.fliplr(hfield)
    elif orientation == "flip_y_up_down":
        hfield = np.flipud(hfield)
    elif orientation == "flip_xy_rotate_180":
        hfield = np.flipud(np.fliplr(hfield))
    elif orientation != "normal":
        raise ValueError(f"Unsupported orientation: {orientation}")

    if manual.shape != hfield.shape:
        raise RuntimeError(
            f"Map and hfield sizes differ: {manual.shape} vs {hfield.shape}"
        )

    manual_occ = manual < 100
    # The corrected OBJ encodes many internal obstacles as low raised geometry,
    # while the outer walls occupy the top of the height range. This threshold
    # keeps the true raised structures but avoids turning the low floor texture
    # into Nav2 obstacles.
    obj_occ = hfield > hfield_threshold
    obj_occ = ndimage.binary_closing(obj_occ, structure=np.ones((3, 3)))
    obj_occ = _remove_small_components(obj_occ, min_obj_area)

    combined = manual_occ | obj_occ
    combined = ndimage.binary_closing(combined, structure=np.ones((3, 3)))

    refined = np.full(manual.shape, 254, dtype=np.uint8)
    refined[combined] = 0

    if backup_pgm is not None and input_pgm.resolve() == output_pgm.resolve():
        backup_pgm.parent.mkdir(parents=True, exist_ok=True)
        if not backup_pgm.exists():
            Image.fromarray(manual, mode="L").save(backup_pgm)

    output_pgm.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(refined, mode="L").save(output_pgm)

    added = int((combined & ~manual_occ).sum())
    removed = int((manual_occ & ~combined).sum())
    print(f"Input: {input_pgm}")
    print(f"Hfield: {hfield_png}")
    print(f"Output: {output_pgm}")
    if backup_pgm is not None:
        print(f"Backup: {backup_pgm}")
    print(f"Hfield threshold: {hfield_threshold}")
    print(f"Orientation: {orientation}")
    print(f"Manual occupied cells: {int(manual_occ.sum())}")
    print(f"Refined occupied cells: {int(combined.sum())}")
    print(f"Added occupied cells from OBJ/hfield: {added}")
    print(f"Removed occupied cells: {removed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hfield", type=Path, default=DEFAULT_HFIELD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--hfield-threshold", type=int, default=15)
    parser.add_argument("--min-obj-area", type=int, default=8)
    parser.add_argument(
        "--orientation",
        choices=ORIENTATIONS,
        default=DEFAULT_ORIENTATION,
        help="Extra transform to apply to the hfield before merging.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup when overwriting the input map.",
    )
    args = parser.parse_args()

    refine_map(
        input_pgm=args.input,
        hfield_png=args.hfield,
        output_pgm=args.output,
        backup_pgm=None if args.no_backup else args.backup,
        hfield_threshold=args.hfield_threshold,
        min_obj_area=args.min_obj_area,
        orientation=args.orientation,
    )


if __name__ == "__main__":
    main()
