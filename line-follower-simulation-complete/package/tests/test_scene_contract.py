from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from ur7e_line_follower.target_line import random_line_from_start, DEFAULT_HOME_DOT
from ur7e_line_follower.trajectory_visual import (
    LINE_MARKERS,
    MODEL_NAME,
    trajectory_model_sdf,
    inject_trajectory_into_world,
)


def test_one_piece_scene_contains_flags_and_line_markers(tmp_path):
    wp = random_line_from_start(np.random.default_rng(1), DEFAULT_HOME_DOT)
    xml = trajectory_model_sdf(wp)
    root = ET.fromstring(xml)
    model = root.find('model')
    assert model is not None and model.attrib['name'] == MODEL_NAME
    visuals = model.find('link').findall('visual')
    names = {v.attrib['name'] for v in visuals}
    assert len(visuals) == LINE_MARKERS + 4
    assert {'pole_start', 'cloth_start', 'pole_end', 'cloth_end'} <= names
    assert 'sph_000' in names and f'sph_{LINE_MARKERS-1:03d}' in names


def test_base_world_does_not_duplicate_dynamic_scene():
    sdf = (Path(__file__).parents[1] / 'worlds' / 'line_follower.sdf').read_text()
    assert 'model name="trajectory_visual"' not in sdf
    assert 'model name="sph_000"' not in sdf
    assert 'model name="pole_start"' not in sdf


def test_runtime_world_injection_is_valid(tmp_path):
    base = Path(__file__).parents[1] / 'worlds' / 'line_follower.sdf'
    out = tmp_path / 'runtime.sdf'
    wp = random_line_from_start(np.random.default_rng(2), DEFAULT_HOME_DOT)
    inject_trajectory_into_world(base, out, wp)
    root = ET.parse(out).getroot()
    world = root.find('world')
    models = [m.attrib.get('name') for m in world.findall('model')]
    assert models.count(MODEL_NAME) == 1
