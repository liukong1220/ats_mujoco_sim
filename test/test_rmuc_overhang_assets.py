"""The scene must load the scale and exact float data generated together."""

import hashlib
import struct
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml


def test_scene_uses_generated_hfield_without_rescaling_it():
    models = Path(__file__).resolve().parents[1] / 'models'
    metadata = yaml.safe_load((models / 'overhangs_2027.generated.yaml').read_text())
    asset = ElementTree.parse(models / 'rmuc_2027_swerve.xml').find('./asset/hfield')
    assert asset.attrib['file'] == 'rmuc_2027_height.bin'
    scale = [float(v) for v in asset.attrib['size'].split()]
    assert scale[:2] == metadata['hfield_half_extents']
    assert scale[2] == metadata['hfield_elevation']
    raw = (models / asset.attrib['file']).read_bytes()
    assert struct.unpack('<ii', raw[:8]) == (metadata['height'], metadata['width'])
    values = np.frombuffer(raw, dtype='<f4', offset=8)
    assert values.min() == pytest.approx(0.)
    assert values.max() == pytest.approx(1.)
    for name, expected in metadata['outputs'].items():
        assert hashlib.sha256((models / name).read_bytes()).hexdigest() == expected
