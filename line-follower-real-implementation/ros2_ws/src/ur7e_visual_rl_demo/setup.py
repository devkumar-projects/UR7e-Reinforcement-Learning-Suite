import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ur7e_visual_rl_demo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dev Kumar',
    maintainer_email='devk79036@gmail.com',
    description='Real UR7e camera + laser line-following demo using SAC, KLT, EKF, calibrated MGD/MGI and LQR filtering.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'visual_detector = ur7e_visual_rl_demo.visual_detector:main',
            'camera_laser_calibrator = ur7e_visual_rl_demo.camera_laser_calibrator:main',
            'observer_probe = ur7e_visual_rl_demo.observer_probe:main',
            'visual_policy_runner = ur7e_visual_rl_demo.visual_policy_runner:main',
            'observer_dashboard = ur7e_visual_rl_demo.observer_dashboard:main',
            'model_check = ur7e_visual_rl_demo.model_check:main',
            'laser_geometry_probe = ur7e_visual_rl_demo.laser_geometry_probe:main',
        ],
    },
)
