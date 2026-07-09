#!/usr/bin/env python3
"""Generate the RMUC 2026 MuJoCo hfield from the corrected field mesh."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


DEFAULT_MAP_YAML = Path("src/ats_sentry_bringup/map/rmuc_2026.yaml")
DEFAULT_MESH = Path("src/ats_sentry_bringup/map/rmuc_2026.stl")
DEFAULT_OUTPUT = Path("src/sim/ats_mujoco_sim/models/rmuc_2026_height.png")
DEFAULT_ORIENTATION = "flip_y_up_down"
ORIENTATIONS = (
    "normal",
    "flip_x_left_right",
    "flip_y_up_down",
    "flip_xy_rotate_180",
)


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                tokens = line.split()[1:4]
                if len(tokens) != 3:
                    continue
                face = tuple(int(token.split("/")[0]) - 1 for token in tokens)
                faces.append(face)

    if not vertices or not faces:
        raise RuntimeError(f"No vertices/faces were parsed from {path}")

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"Binary STL is too small: {path}")

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise RuntimeError(
            f"Binary STL size mismatch for {path}: expected {expected_size}, got {len(data)}"
        )

    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    triangles = np.frombuffer(data, dtype=dtype, offset=84, count=triangle_count)
    vertices = triangles["vertices"].reshape(-1, 3).astype(np.float64)
    faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    if vertices.size == 0:
        raise RuntimeError(f"No triangles were parsed from {path}")
    return vertices, faces


def _read_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as stl_file:
        for line in stl_file:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))

    if len(vertices) < 3:
        raise RuntimeError(f"No vertices were parsed from ASCII STL {path}")

    usable_count = len(vertices) - (len(vertices) % 3)
    vertices_array = np.asarray(vertices[:usable_count], dtype=np.float64)
    faces = np.arange(usable_count, dtype=np.int32).reshape(-1, 3)
    return vertices_array, faces


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stl_file:
        header = stl_file.read(84)

    if len(header) >= 84:
        triangle_count = struct.unpack_from("<I", header, 80)[0]
        if 84 + triangle_count * 50 == path.stat().st_size:
            return _read_binary_stl(path)
    return _read_ascii_stl(path)


def _read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _read_obj(path)
    if suffix == ".stl":
        return _read_stl(path)
    raise ValueError(f"Unsupported mesh format for {path}; expected .stl or .obj")


def _load_map_metadata(map_yaml: Path) -> tuple[int, int, float, tuple[float, float]]:
    with map_yaml.open("r", encoding="utf-8") as yaml_file:
        metadata = yaml.safe_load(yaml_file)

    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    image = Image.open(image_path)
    width, height = image.size
    origin = metadata["origin"]

    return width, height, float(metadata["resolution"]), (float(origin[0]), float(origin[1]))


def _load_map_image(map_yaml: Path) -> np.ndarray:
    with map_yaml.open("r", encoding="utf-8") as yaml_file:
        metadata = yaml.safe_load(yaml_file)

    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    return np.asarray(Image.open(image_path).convert("L"))


def _rasterize_triangle(
    hfield: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> None:
    height, width = hfield.shape

    min_col = max(0, int(math.floor(float(np.min(px)))))
    max_col = min(width - 1, int(math.ceil(float(np.max(px)))))
    min_row = max(0, int(math.floor(float(np.min(py)))))
    max_row = min(height - 1, int(math.ceil(float(np.max(py)))))
    if min_col > max_col or min_row > max_row:
        return

    x0, x1, x2 = px
    y0, y1, y2 = py
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1.0e-9:
        return

    xs = np.arange(min_col, max_col + 1, dtype=np.float64)
    ys = np.arange(min_row, max_row + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    eps = 1.0e-7
    inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
    if not np.any(inside):
        return

    z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
    patch = hfield[min_row : max_row + 1, min_col : max_col + 1]
    np.maximum(patch, np.where(inside, z, -np.inf), out=patch)


def generate_hfield(
    mesh_path: Path,
    map_yaml: Path,
    output_path: Path,
    z_scale: float,
    ground_percentile: float,
    orientation: str,
    robot_clearance_height: float,
    clear_nav_free_space: bool,
) -> float:
    width, height, resolution, origin = _load_map_metadata(map_yaml)
    nav_map = _load_map_image(map_yaml)
    if nav_map.shape != (height, width):
        raise RuntimeError(
            f"Map image shape {nav_map.shape} does not match YAML size {(height, width)}"
        )
    vertices, faces = _read_mesh(mesh_path)

    # Keep the hfield centered exactly like the MuJoCo XML. The YAML origin is
    # still used for image dimensions/resolution so RViz and MuJoCo stay aligned.
    half_x = width * resolution * 0.5
    half_y = height * resolution * 0.5

    z_ground = float(np.percentile(vertices[:, 2], ground_percentile))
    scaled_z = np.maximum(0.0, (vertices[:, 2] - z_ground) * z_scale)
    clearance_mesh_z = z_ground + robot_clearance_height / z_scale

    px = (vertices[:, 0] + half_x) / (2.0 * half_x) * (width - 1)
    py = (vertices[:, 1] + half_y) / (2.0 * half_y) * (height - 1)

    hfield = np.zeros((height, width), dtype=np.float32)
    tri_vertices = np.column_stack((px, py, scaled_z))
    skipped_overhead_faces = 0
    for face in faces:
        # MuJoCo hfields are single-valued: overhead beams would otherwise be
        # projected down as solid walls. Keep only geometry that the 23 cm robot
        # body can actually collide with.
        if float(np.min(vertices[face, 2])) > clearance_mesh_z:
            skipped_overhead_faces += 1
            continue
        tri = tri_vertices[face]
        _rasterize_triangle(hfield, tri[:, 0], tri[:, 1], tri[:, 2])

    elevation = float(np.max(hfield))
    if elevation <= 0.0:
        raise RuntimeError("Generated hfield is empty; check mesh/map alignment")

    image = np.clip(np.rint(hfield / elevation * 255.0), 0, 255).astype(np.uint8)
    if orientation == "flip_x_left_right":
        image = np.fliplr(image)
    elif orientation == "flip_y_up_down":
        image = np.flipud(image)
    elif orientation == "flip_xy_rotate_180":
        image = np.flipud(np.fliplr(image))
    elif orientation != "normal":
        raise ValueError(f"Unsupported orientation: {orientation}")

    cleared_free_cells = 0
    if clear_nav_free_space:
        # The STL may contain overhead beams or side faces that a single-valued
        # MuJoCo hfield would project down into passable tunnels. The final Nav2
        # PGM is the authoritative 2D traversability mask, so keep STL height
        # only on occupied / boundary cells and flatten white free space.
        free_mask = nav_map >= 200
        cleared_free_cells = int(np.count_nonzero((image > 0) & free_mask))
        image[free_mask] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path)

    print(f"Mesh: {mesh_path}")
    print(f"Map: {map_yaml}")
    print(f"Output: {output_path}")
    print(f"Image: {width}x{height}")
    print(f"Half extents: x={half_x:.6f}, y={half_y:.6f}")
    print(f"Ground z percentile {ground_percentile:g}: {z_ground:.6f}")
    print(f"Z scale: {z_scale:.6f}")
    print(f"Robot clearance height: {robot_clearance_height:.6f} m")
    print(f"Skipped overhead faces: {skipped_overhead_faces}")
    print(f"Clear Nav2 free space: {clear_nav_free_space}")
    print(f"Cleared free hfield cells: {cleared_free_cells}")
    print(f"Orientation: {orientation}")
    print(f"MuJoCo hfield elevation: {elevation:.6f}")
    return elevation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument(
        "--obj",
        type=Path,
        default=None,
        help="Deprecated alias for --mesh, kept for older commands.",
    )
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_MAP_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--z-scale", type=float, default=0.1)
    parser.add_argument("--ground-percentile", type=float, default=0.0)
    parser.add_argument(
        "--robot-clearance-height",
        type=float,
        default=0.28,
        help=(
            "Ignore mesh faces whose lowest point is above this real-world "
            "height, so overhead limit beams do not become solid hfield walls."
        ),
    )
    parser.add_argument(
        "--no-clear-nav-free-space",
        action="store_true",
        help="Keep raw STL heights in white/free PGM cells.",
    )
    parser.add_argument(
        "--orientation",
        choices=ORIENTATIONS,
        default=DEFAULT_ORIENTATION,
        help="Projection orientation relative to the Nav2 occupancy image.",
    )
    args = parser.parse_args()
    mesh_path = args.obj if args.obj is not None else args.mesh

    generate_hfield(
        mesh_path=mesh_path,
        map_yaml=args.map_yaml,
        output_path=args.output,
        z_scale=args.z_scale,
        ground_percentile=args.ground_percentile,
        orientation=args.orientation,
        robot_clearance_height=args.robot_clearance_height,
        clear_nav_free_space=not args.no_clear_nav_free_space,
    )


if __name__ == "__main__":
    main()
