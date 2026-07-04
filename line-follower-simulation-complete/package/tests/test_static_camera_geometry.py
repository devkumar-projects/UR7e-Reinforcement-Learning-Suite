"""Validation géométrique hors Gazebo du placement de la caméra statique."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _camera_data():
    world = Path(__file__).resolve().parents[1] / 'worlds' / 'line_follower.sdf'
    root = ET.parse(world).getroot()
    model = root.find(".//model[@name='static_line_camera']")
    assert model is not None
    link = model.find("link[@name='camera_link']")
    assert link is not None
    pose = np.array([float(v) for v in link.findtext('pose').split()])
    sensor = link.find("sensor[@name='line_camera']")
    hfov = float(sensor.findtext('camera/horizontal_fov'))
    width = int(sensor.findtext('camera/image/width'))
    height = int(sensor.findtext('camera/image/height'))
    return pose, hfov, width, height


def test_camera_points_toward_wall_center():
    pose, _, _, _ = _camera_data()
    position = pose[:3]
    R = _rpy(*pose[3:])
    forward = R @ np.array([1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.75]) - position
    target /= np.linalg.norm(target)
    assert float(np.dot(forward, target)) > 0.999


def test_useful_wall_area_fits_in_field_of_view():
    pose, hfov, width, height = _camera_data()
    position = pose[:3]
    R = _rpy(*pose[3:])
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * height / width)
    for y in (-0.65, 0.65):
        for z in (0.20, 1.30):
            pc = R.T @ (np.array([1.0, y, z]) - position)
            assert pc[0] > 0.0
            h = abs(math.atan2(pc[1], pc[0]))
            v = abs(math.atan2(pc[2], pc[0]))
            assert h < hfov / 2.0
            assert v < vfov / 2.0
