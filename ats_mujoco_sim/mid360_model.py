"""Helpers for sharing the MID360 mesh with generated MuJoCo scenes."""

from __future__ import annotations

from pathlib import Path
import shutil


MID360_MESH_RELATIVE_PATH = "meshes/mid360.obj"
MID360_MESH_PACKAGE_PATH = "models/meshes/mid360.obj"


def resolve_mid360_mesh_path() -> Path | None:
    """Resolve the MuJoCo-readable MID360 mesh."""
    candidates: list[Path] = []

    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("ats_mujoco_sim")) / MID360_MESH_PACKAGE_PATH)
    except Exception:
        pass

    package_file = Path(__file__).resolve()
    parents = package_file.parents
    if len(parents) > 3:
        candidates.append(parents[1] / MID360_MESH_PACKAGE_PATH)
    if len(parents) > 4:
        candidates.append(
            parents[4]
            / "src"
            / "sim"
            / "ats_mujoco_sim"
            / MID360_MESH_PACKAGE_PATH
        )
        candidates.append(
            parents[4]
            / "ats_mujoco_sim"
            / "share"
            / "ats_mujoco_sim"
            / MID360_MESH_PACKAGE_PATH
        )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    return None


def copy_mid360_mesh_assets(model_dir: Path) -> Path | None:
    """Copy the MuJoCo-readable MID360 mesh into a scene model directory."""
    mesh_path = resolve_mid360_mesh_path()
    if mesh_path is None:
        return None
    output_path = Path(model_dir) / MID360_MESH_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mesh_path.resolve() != output_path.resolve():
        shutil.copy2(mesh_path, output_path)
    return output_path
