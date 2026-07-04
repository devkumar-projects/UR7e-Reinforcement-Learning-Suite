"""Generate the complete flags + blue trajectory as one Gazebo model.

The first scene is injected into the runtime world before Gazebo starts, so no
transport service is required to make the drawing visible. Later drawings can
replace the same single model through Gazebo's create/remove services.
"""
from __future__ import annotations

from pathlib import Path
import html
import numpy as np

MODEL_NAME = 'trajectory_visual'
LINE_MARKERS = 50
LINE_X = 0.990
FLAG_X = 0.985
FLAG_POLE_H = 0.220
FLAG_CLOTH_H = 0.055
FLAG_CLOTH_W = 0.090


def resample_waypoints(waypoints: np.ndarray, n: int = LINE_MARKERS) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        raise ValueError('waypoints must have shape (N,2), N>=2')
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError('trajectory length is zero')
    s = np.linspace(0.0, total, int(n))
    y = np.interp(s, cum, wp[:, 0])
    z = np.interp(s, cum, wp[:, 1])
    return np.column_stack((y, z))


def _visual(name: str, pose: str, geometry: str, material: str) -> str:
    return f'''      <visual name="{name}">
        <pose>{pose}</pose>
        <geometry>{geometry}</geometry>
        <material>{material}</material>
      </visual>'''


def trajectory_model_sdf(waypoints: np.ndarray, model_name: str = MODEL_NAME) -> str:
    """Return an SDF document containing flags and line in one static model."""
    wp = np.asarray(waypoints, dtype=np.float64)
    pts = resample_waypoints(wp)
    sy, sz = map(float, wp[0])
    ey, ez = map(float, wp[-1])

    dark = '<ambient>0.18 0.18 0.18 1</ambient><diffuse>0.18 0.18 0.18 1</diffuse>'
    red = ('<ambient>1.0 0.04 0.04 1</ambient><diffuse>1.0 0.04 0.04 1</diffuse>'
           '<emissive>0.50 0.02 0.02 1</emissive>')
    green = ('<ambient>0.04 0.88 0.12 1</ambient><diffuse>0.04 0.88 0.12 1</diffuse>'
             '<emissive>0.02 0.44 0.06 1</emissive>')
    blue = ('<ambient>0.05 0.15 0.90 1</ambient><diffuse>0.05 0.18 0.95 1</diffuse>'
            '<emissive>0.01 0.04 0.35 1</emissive>')

    visuals = []
    for y, z, suffix, cloth_mat in ((sy, sz, 'start', red), (ey, ez, 'end', green)):
        visuals.append(_visual(
            f'pole_{suffix}',
            f'{FLAG_X:.5f} {y:.5f} {z + FLAG_POLE_H / 2.0:.5f} 0 0 0',
            f'<cylinder><radius>0.005</radius><length>{FLAG_POLE_H:.5f}</length></cylinder>',
            dark,
        ))
        visuals.append(_visual(
            f'cloth_{suffix}',
            f'{FLAG_X:.5f} {y + FLAG_CLOTH_W / 2.0 + 0.006:.5f} '
            f'{z + FLAG_POLE_H - FLAG_CLOTH_H / 2.0:.5f} 0 0 0',
            f'<box><size>0.005 {FLAG_CLOTH_W:.5f} {FLAG_CLOTH_H:.5f}</size></box>',
            cloth_mat,
        ))

    for i, (y, z) in enumerate(pts):
        visuals.append(_visual(
            f'sph_{i:03d}',
            f'{LINE_X:.5f} {float(y):.5f} {float(z):.5f} 0 0 0',
            '<sphere><radius>0.0250</radius></sphere>',
            blue,
        ))

    body = '\n'.join(visuals)
    return f'''<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{html.escape(model_name, quote=True)}">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
{body}
    </link>
  </model>
</sdf>
'''


def write_trajectory_model(path: str | Path, waypoints: np.ndarray,
                           model_name: str = MODEL_NAME) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trajectory_model_sdf(waypoints, model_name), encoding='utf-8')
    return out


def inject_trajectory_into_world(base_world: str | Path, output_world: str | Path,
                                 waypoints: np.ndarray) -> Path:
    """Inject the one-piece trajectory model immediately before </world>."""
    base = Path(base_world).read_text(encoding='utf-8')
    marker = '</world>'
    if marker not in base:
        raise ValueError(f'{base_world}: missing </world>')
    model_doc = trajectory_model_sdf(waypoints)
    model_xml = model_doc.split('<sdf version="1.8">', 1)[1].rsplit('</sdf>', 1)[0].strip()
    runtime = base.replace(marker, f'\n    {model_xml}\n  {marker}', 1)
    out = Path(output_world)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(runtime, encoding='utf-8')
    return out
