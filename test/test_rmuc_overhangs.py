"""Small deterministic multi-layer generation tests (no official mesh required)."""

import numpy as np
import pytest

from ats_mujoco_sim.rmuc_overhangs import _box_geometry, partition_layers, roof_xml


def test_two_axis_plane_and_thickness_are_preserved():
    cells = [[y * 8 + x, .1, .4 + .02 * x - .03 * y, .45 + .02 * x - .03 * y]
             for y in range(4) for x in range(5)]
    plates = partition_layers(cells, 8, 1.)
    assert len(plates) == 1
    assert plates[0]['slope_x'] == pytest.approx(.02)
    assert plates[0]['slope_y'] == pytest.approx(-.03)
    assert 'quat=' in roof_xml(plates)


def test_holes_and_low_lips_are_not_filled_or_dropped():
    cells = [[0, .13, .22, .5], [1, .13, .22, .5], [4, .13, .22, .5]]
    plates = partition_layers(cells, 3, .02)
    assert sum(p['cells'] for p in plates) == 3
    assert sum((p['bounds_xy'][1] - p['bounds_xy'][0]) *
               (p['bounds_xy'][3] - p['bounds_xy'][2]) for p in plates) == pytest.approx(3*.02**2)


def test_curved_or_nonparallel_plate_recursively_splits():
    cells = np.array([[y * 8 + x, .1, .4 + .002 * x*x, .5 + .003 * y*y]
                      for y in range(5) for x in range(5)])
    plates = partition_layers(cells, 8, .02)
    assert len(plates) > 1
    assert sum(p['cells'] for p in plates) == 25
    assert all(p['bottom_error_m'] <= .003 and p['top_error_m'] <= .005 for p in plates)


def test_invalid_layers_fail_closed():
    with pytest.raises(ValueError):
        partition_layers([[0, .2, .1, .5]], 4, .02)
    with pytest.raises(ValueError):
        partition_layers([[0, .1, .3, .4], [0, .1, .3, .4]], 4, .02)


def test_steep_two_axis_box_does_not_shadow_neighbouring_air():
    # Full-field v2 failure: the orthogonal bounding box of a two-axis plane
    # extends farther than the thickness-times-slope estimate alone predicts.
    cells = [[y*8+x, .326, .73-.4375*x*.02-.9375*y*.02,
              .734-.4375*x*.02-.9375*y*.02] for y in range(2) for x in range(2)]
    plates = partition_layers(cells, 8, .02)
    for p in plates:
        rotation, half = _box_geometry(p['bounds_xy'], p['bottom_z'], p['top_z'],
                                        p['slope_x'], p['slope_y'])
        x0, x1, y0, y1 = p['bounds_xy']
        assert np.all(np.abs(rotation[:2]) @ half <= np.array([x1-x0, y1-y0])/2 + .0025)
