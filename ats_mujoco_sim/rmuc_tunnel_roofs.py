"""Generate MuJoCo tunnel roof collision boxes from tunnels_2027.yaml."""

from pathlib import Path
import argparse
import yaml


def generate_roof_xml(config_path: str) -> str:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    bodies = []
    for tunnel in config.get("tunnels", []):
        polygon = tunnel["polygon"]
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        bottom = float(tunnel["roof_bottom_z"])
        top = float(tunnel["roof_top_z"])
        if top <= bottom:
            raise ValueError(f"{tunnel['name']}: roof_top_z must exceed roof_bottom_z")
        # 2027 当前多边形为轴对齐矩形；非矩形输入先 fail closed，避免生成错误碰撞盒。
        if len(polygon) != 4 or max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
            raise ValueError(f"{tunnel['name']}: polygon must be a non-empty rectangle")
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        bodies.append(
            f'  <geom name="roof_{tunnel["name"]}" type="box" '
            f'pos="{cx:.6f} {cy:.6f} {(bottom + top) / 2.0:.6f}" '
            f'size="{(max(xs) - min(xs)) / 2.0:.6f} {(max(ys) - min(ys)) / 2.0:.6f} '
            f'{(top - bottom) / 2.0:.6f}" rgba="0.2 0.4 0.9 0.35" '
            'contype="1" conaffinity="1" friction="1.2 0.08 0.02" group="3"/> '
        )
    return '<!-- Generated from tunnels_2027.yaml. Do not edit by hand. -->\n' \
        '<body name="rmuc_2027_tunnel_roofs" pos="0 0 0">\n' \
        + "\n".join(bodies) + '\n</body>\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tunnels-yaml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(generate_roof_xml(args.tunnels_yaml), encoding="utf-8")


if __name__ == "__main__":
    main()
