"""
Test ROS2 : CameraLineDetector publie bien 7 valeurs (schéma V4).
Sauté si rclpy n'est pas disponible.

Vérifie :
  - réception d'un message sur /line_detection après injection d'une Image sur /line_camera
  - longueur du vecteur == 7 (schéma V4)
  - toutes les valeurs sont finies
  - indices conformes au schéma V4 (detected ∈ {0,1}, laser_visible ∈ {0,1})
  - arrêt propre des nœuds même en cas d'échec (try/finally)
"""
import sys
import pathlib
import numpy as np
import cv2
import pytest

_pkg_root = pathlib.Path(__file__).resolve().parents[1]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

rclpy = pytest.importorskip('rclpy')


def make_image_msg(width=320, height=240):
    from sensor_msgs.msg import Image
    msg = Image()
    msg.width = width
    msg.height = height
    msg.encoding = 'bgr8'
    msg.step = width * 3
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Primitives épaisses afin de survivre au filtre morphologique 3x3.
    cv2.line(frame, (20, height // 2), (width - 20, height // 2),
             (200, 20, 10), thickness=7)
    cv2.circle(frame, (width // 2, height // 2 + 20), 6,
               (20, 20, 200), thickness=-1)
    msg.data = frame.tobytes()
    return msg


def test_line_detector_publishes_v4_schema():
    """
    line_detector publie Float32MultiArray avec 7 éléments en schéma V4.
    Vérifie longueur, valeurs finies, et conformité des indices.
    Nœuds toujours détruits en fin de test (try/finally).
    """
    import time
    from std_msgs.msg import Float32MultiArray
    from sensor_msgs.msg import Image
    from ur7e_line_follower.camera_line_detector import (
        CameraLineDetector, OBSERVATION_SCHEMA_VERSION,
    )

    print(f"  OBSERVATION_SCHEMA_VERSION={OBSERVATION_SCHEMA_VERSION}")

    if not rclpy.ok():
        rclpy.init()

    received = []
    guidance_received = []
    subscriber_node = None
    detector = None
    executor = None

    try:
        subscriber_node = rclpy.create_node('test_sub_v4')
        subscriber_node.create_subscription(
            Float32MultiArray, '/line_detection',
            lambda msg: received.append(list(msg.data)), 10,
        )
        subscriber_node.create_subscription(
            Float32MultiArray, '/line_guidance',
            lambda msg: guidance_received.append(list(msg.data)), 10,
        )

        from rclpy.qos import qos_profile_sensor_data
        detector = CameraLineDetector()
        pub = subscriber_node.create_publisher(
            Image, '/line_camera', qos_profile_sensor_data)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(detector)
        executor.add_node(subscriber_node)

        img_msg = make_image_msg()
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and len(received) < 3:
            pub.publish(img_msg)
            executor.spin_once(timeout_sec=0.05)

    finally:
        if executor is not None:
            executor.shutdown()
        if detector is not None:
            try: detector.destroy_node()
            except Exception: pass
        if subscriber_node is not None:
            try: subscriber_node.destroy_node()
            except Exception: pass

    assert len(received) >= 1, "Aucun message reçu sur /line_detection"
    assert len(guidance_received) >= 1, "Aucun message reçu sur /line_guidance"
    assert len(guidance_received[-1]) == 3
    assert all(np.isfinite(v) and -1.0 <= v <= 1.0 for v in guidance_received[-1])

    for i, data in enumerate(received):
        # Longueur V2
        assert len(data) == 7, \
            f"msg#{i}: detection_vector a {len(data)} valeurs, attendu 7 (schéma V4)"
        # Toutes finies
        assert all(np.isfinite(v) for v in data), \
            f"msg#{i}: valeurs non-finies: {data}"
        # detected ∈ {{0.0, 1.0}}
        assert data[0] in (0.0, 1.0), f"msg#{i}: data[0] (detected)={data[0]}"
        # laser_visible ∈ {{0.0, 1.0}}
        assert data[6] in (0.0, 1.0), f"msg#{i}: data[6] (laser_visible)={data[6]}"
        # offset_n, klt_confidence, cos_2t, sin_2t, coverage ∈ [-1, 1]
        for idx in [1, 2, 3, 4, 5]:
            assert -1.0 - 1e-4 <= data[idx] <= 1.0 + 1e-4, \
                f"msg#{i}: data[{idx}]={data[idx]} hors [-1,1]"
        print(f"  msg#{i}: {[round(v,3) for v in data]}")
