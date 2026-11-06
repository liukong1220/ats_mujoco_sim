"""Generate MuJoCo tunnel roof collision boxes from tunnels_2027.yaml."""

from pathlib import Path
import argparse
import math
import yaml


def generate_roof_xml(config_path: str) -> str:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # 隧道顶板与 overhangs: 里的悬空板都不进 hfield，物理碰撞都由这里的盒子提供。
    # bottom_z / top_z 取矩形中心，slope_y = dz/dy 让盒子绕 x 轴倾斜；隧道顶板与口袋底平行，
    # 坡度取 roof_slope_y（缺省水平）。
    plates = [("roof_" + t["name"], t["polygon"], t["roof_bottom_z"], t["roof_top_z"],
               float(t.get("roof_slope_y", 0.0)))
              for t in config.get("tunnels", [])]
    plates += [("overhang_" + o["name"], o["polygon"], o["bottom_z"], o["top_z"],
                float(o.get("slope_y", 0.0))) for o in config.get("overhangs", [])]
    bodies = []
    for name, polygon, bottom, top, slope_y in plates:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        bottom = float(bottom)
        top = float(top)
        if top <= bottom:
            raise ValueError(f"{name}: top must exceed bottom")
        # 2027 当前多边形为轴对齐矩形；非矩形输入先 fail closed，避免生成错误碰撞盒。
        if len(polygon) != 4 or max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
            raise ValueError(f"{name}: polygon must be a non-empty rectangle")
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        # 绕 x 轴转 tilt 后，盒子局部 y 半长按 1/cos 拉长才能盖住投影矩形，法向厚度是竖直厚度乘 cos。
        # 用 quat 而不是 euler，避免依赖 <compiler angle>。
        tilt = math.atan(slope_y)
        quat_attr = (f'quat="{math.cos(tilt / 2.0):.6f} {math.sin(tilt / 2.0):.6f} 0 0" '
                     if slope_y != 0.0 else '')
        bodies.append(
            f'  <geom name="{name}" type="box" '
            f'pos="{cx:.6f} {cy:.6f} {(bottom + top) / 2.0:.6f}" {quat_attr}'
            f'size="{(max(xs) - min(xs)) / 2.0:.6f} '
            f'{(max(ys) - min(ys)) / 2.0 / math.cos(tilt):.6f} '
            f'{(top - bottom) * math.cos(tilt) / 2.0:.6f}" rgba="0.2 0.4 0.9 0.35" '
            'contype="1" conaffinity="1" friction="1.2 0.08 0.02" group="3"/>'
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
