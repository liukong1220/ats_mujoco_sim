from pathlib import Path

import numpy as np
from PIL import Image

from ats_mujoco_sim.static_map_publisher import load_static_occupancy_grid


def test_trinary_map_preserves_origin_and_flips_image_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "test_map.pgm"
    Image.fromarray(
        np.array([[0, 127], [254, 255]], dtype=np.uint8), mode="L"
    ).save(image_path)
    yaml_path = tmp_path / "test_map.yaml"
    yaml_path.write_text(
        "\n".join([
            "image: test_map.pgm",
            "resolution: 0.2",
            "origin: [1.5, -2.0, 0.0]",
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.25",
        ]),
        encoding="utf-8",
    )

    grid = load_static_occupancy_grid(yaml_path, "map")

    assert grid.header.frame_id == "map"
    assert grid.info.width == 2
    assert grid.info.height == 2
    assert grid.info.resolution == 0.2
    assert grid.info.origin.position.x == 1.5
    assert grid.info.origin.position.y == -2.0
    # 图像最下行映射为 ROS 栅格首行：free、free、occupied、unknown。
    assert list(grid.data) == [0, 0, 100, -1]
