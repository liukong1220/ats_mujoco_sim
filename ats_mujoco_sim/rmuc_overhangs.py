"""Collision generation from audited vertical layers, never navigation input.

No positive cavity-fill threshold has been proved for the complete wheel/body
assembly. Consequently g_box is zero: even low lips retain their air layer.
"""

from __future__ import annotations

import numpy as np


def _box_geometry(bounds, bottom, top, b, c):
    normal = np.array([-b, -c, 1.])
    normal /= np.linalg.norm(normal)
    tx = np.array([1., 0., b])
    tx /= np.linalg.norm(tx)
    ty = np.cross(normal, tx)
    rotation = np.column_stack([tx, ty, normal])
    x0, x1, y0, y1 = bounds
    cx, cy = (x0+x1)/2, (y0+y1)/2
    corners = np.array([[x-cx, y-cy, b*(x-cx)+c*(y-cy)]
                        for x in (x0, x1) for y in (y0, y1)])
    half = np.max(np.abs(corners @ rotation), axis=0)
    half[2] = (top-bottom) * normal[2] / 2
    return rotation, half


def partition_layers(cells, width, step, bottom_error=.003, top_error=.005):
    """Recursively tile occupied samples; never bridge holes in a cluster.

    Each leaf has parallel lower/upper fitted planes. Nonparallel or curved
    surfaces split until both error bounds hold. Single-cell leaves remain
    explicit, rather than silently dropping complex boundaries.
    """
    cells = np.asarray(cells, dtype=float)
    if cells.ndim != 2 or cells.shape[1] < 4 or not np.isfinite(cells[:, :4]).all():
        raise ValueError('expected finite column/floor/bottom/top records')
    if np.any(cells[:, 3] <= cells[:, 2]) or np.any(cells[:, 2] <= cells[:, 1]):
        raise ValueError('invalid layer ordering')
    ids = cells[:, 0].astype(int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate columns')
    cols, rows = ids % width, ids // width
    output = []

    def split(indices):
        cc, rr = cols[indices], rows[indices]
        x0, x1, y0, y1 = cc.min(), cc.max(), rr.min(), rr.max()
        x, y = (cc + .5) * step, (rr + .5) * step
        cx, cy = (x0 + x1 + 1) * step / 2, (y0 + y1 + 1) * step / 2
        design = np.c_[np.ones(len(indices)), x - cx, y - cy]
        bottom = cells[indices, 2]
        fit = np.linalg.lstsq(design, bottom, rcond=None)[0]
        residual = bottom - design @ fit
        # Both surfaces use the same normal; varying thickness triggers a split.
        top_offsets = cells[indices, 3] - design[:, 1:] @ fit[1:]
        top = (top_offsets.min() + top_offsets.max()) / 2
        be = float(np.max(np.abs(residual)))
        te = float(np.max(np.abs(top_offsets - top)))
        # Thick, steep boxes can project far beyond their audited rectangle.
        # Split them; at a single sample use a vertical-sided column instead.
        # This prevents a neighbouring steep roof from shadowing valid air.
        bounds = [x0*step, (x1+1)*step, y0*step, (y1+1)*step]
        rotation, half = _box_geometry(bounds, fit[0], top, *fit[1:])
        xy_radius = np.abs(rotation[:2]) @ half
        overreach = np.max(xy_radius - [(x1-x0+1)*step/2, (y1-y0+1)*step/2])
        if len(indices) == 1:
            fit = np.array([bottom[0], 0., 0.])
            top = cells[indices[0], 3]
            be = te = overreach = 0.
        full = len(indices) == (x1 - x0 + 1) * (y1 - y0 + 1)
        if full and be <= bottom_error and te <= top_error and overreach < step / 8:
            output.append(dict(bounds_xy=[float(x0 * step), float((x1 + 1) * step),
                                          float(y0 * step), float((y1 + 1) * step)],
                               center_xy=[float(cx), float(cy)], bottom_z=float(fit[0]),
                               top_z=float(top), slope_x=float(fit[1]),
                               slope_y=float(fit[2]), cells=len(indices),
                               bottom_error_m=be, top_error_m=te))
            return
        if x1 - x0 >= y1 - y0:
            left = cc <= (x0 + x1) // 2
        else:
            left = rr <= (y0 + y1) // 2
        for selection in (indices[left], indices[~left]):
            if len(selection):
                split(selection)

    if len(cells):
        split(np.arange(len(cells)))
    return output


def roof_xml(plates):
    """Two-axis tilted boxes; quat uses MuJoCo's w,x,y,z convention."""
    from scipy.spatial.transform import Rotation

    geoms = []
    for index, plate in enumerate(plates):
        b, c = plate['slope_x'], plate['slope_y']
        rotation, half = _box_geometry(plate['bounds_xy'], plate['bottom_z'], plate['top_z'], b, c)
        cx, cy = plate['center_xy']
        mid = (plate['bottom_z'] + plate['top_z']) / 2
        # Sub-micron seam closure avoids a floating-point no-hit at the exact
        # shared edge of two boxes (e.g. the D1 probe y=1.500000).
        half[:2] += 1.e-7
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        values = lambda a: ' '.join(f'{v:.9f}' for v in a)
        geoms.append(f'  <geom name="audited_roof_{index}" type="box" '
                     f'pos="{values([cx, cy, mid])}" quat="{values(quat)}" '
                     f'size="{values(half)}" rgba="0.2 0.4 0.9 0.35" '
                     'contype="1" conaffinity="1" friction="1.2 0.08 0.02" group="3"/>')
    return ('<!-- Generated from official-mesh audit; do not edit. -->\n'
            '<body name="rmuc_2027_tunnel_roofs">\n' + '\n'.join(geoms) + '\n</body>\n')
