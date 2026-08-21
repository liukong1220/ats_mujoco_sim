#!/usr/bin/env python3
"""Generate compact MuJoCo collision boxes from the authoritative RMUC map."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


DEFAULT_MAP_YAML = Path("src/ats_sentry_bringup/map/rmuc_2025.yaml")
DEFAULT_OUTPUT = Path("src/sim/ats_mujoco_sim/models/rmuc_2025_wall_boxes.xml")


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int


def _load_map(map_yaml: Path) -> tuple[np.ndarray, float, tuple[float, float]]:
    metadata = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    image = np.asarray(Image.open(image_path).convert("L"))
    occupied = image < int(round(float(metadata["occupied_thresh"]) * 255.0))
    origin = metadata["origin"]
    return occupied, float(metadata["resolution"]), (float(origin[0]), float(origin[1]))


def decompose_rectangles(occupied: np.ndarray) -> list[Rectangle]:
    """Exactly cover a binary map with maximal top-left greedy rectangles."""
    remaining = occupied.copy()
    height, width = remaining.shape
    rectangles: list[Rectangle] = []
    while bool(remaining.any()):
        y, x = np.argwhere(remaining)[0]
        run_width = 0
        while x + run_width < width and remaining[y, x + run_width]:
            run_width += 1
        run_height = 1
        while y + run_height < height and bool(
            remaining[y + run_height, x : x + run_width].all()
        ):
            run_height += 1
        remaining[y : y + run_height, x : x + run_width] = False
        rectangles.append(Rectangle(int(x), int(y), run_width, run_height))
    return rectangles


def build_wall_boxes_xml(
    rectangles: list[Rectangle],
    image_height: int,
    resolution: float,
    origin: tuple[float, float],
    wall_height_m: float,
    scene_name: str = "rmuc_2025",
) -> str:
    half_height = 0.5 * wall_height_m
    lines = [
        f"<!-- Generated from the same {scene_name}.pgm used by static_map_publisher. -->",
        f"<!-- rectangles={len(rectangles)} wall_height_m={wall_height_m:.3f} -->",
        f'    <body name="{scene_name}_static_walls" pos="0 0 0">',
    ]
    for index, rectangle in enumerate(rectangles):
        size_x = 0.5 * rectangle.width * resolution
        size_y = 0.5 * rectangle.height * resolution
        center_x = origin[0] + (rectangle.x + 0.5 * rectangle.width) * resolution
        center_y = origin[1] + (
            image_height - rectangle.y - 0.5 * rectangle.height
        ) * resolution
        lines.append(
            "      "
            f'<geom name="{scene_name}_wall_{index:04d}" type="box" '
            f'pos="{center_x:.4f} {center_y:.4f} {half_height:.4f}" '
            f'size="{size_x:.4f} {size_y:.4f} {half_height:.4f}" '
            'rgba="0.72 0.24 0.10 0.16" friction="1.2 0.08 0.02" '
            'contype="1" conaffinity="1" group="3"/>'
        )
    lines.append("    </body>")
    return "\n".join(lines) + "\n"


def generate_wall_collisions(
    map_yaml: Path,
    output: Path,
    wall_height_m: float = 1.20,
    scene_name: str = "rmuc_2025",
) -> int:
    occupied, resolution, origin = _load_map(map_yaml)
    rectangles = decompose_rectangles(occupied)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_wall_boxes_xml(
            rectangles,
            occupied.shape[0],
            resolution,
            origin,
            wall_height_m,
            scene_name,
        ),
        encoding="utf-8",
    )
    return len(rectangles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wall-height-m", type=float, default=1.20)
    parser.add_argument("--scene-name", default="rmuc_2025")
    args = parser.parse_args()
    count = generate_wall_collisions(
        args.map_yaml, args.output, args.wall_height_m, args.scene_name
    )
    print(f"Generated {count} RMUC collision boxes: {args.output}")


if __name__ == "__main__":
    main()
