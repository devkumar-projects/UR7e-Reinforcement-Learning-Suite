from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur7e_line_follower'

setup(
    name=package_name,
    version='3.3.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'requirements.txt'] + glob('*.md')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'urdf'),  glob('urdf/*')),
    ],
    install_requires=['setuptools', 'numpy', 'gymnasium', 'opencv-python', 'matplotlib', 'pandas', 'stable-baselines3', 'tensorboard', 'rich'],
    zip_safe=True,
    maintainer='Dev Kumar',
    maintainer_email='devk79036@gmail.com',
    description='RL line-following for UR7e — laser pointer traces a hand-drawn line on a wall',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'train         = ur7e_line_follower.train:main',
            'metrics       = ur7e_line_follower.metrics:main',
            'line_detector = ur7e_line_follower.camera_line_detector:main',
            'plot_monitor  = ur7e_line_follower.plot_monitor:main',
            'live_dashboard  = ur7e_line_follower.live_dashboard:main',
            'eval_full       = ur7e_line_follower.eval_full:main',
            'schema_bloc     = ur7e_line_follower.schema_bloc:main',
            'analyse_commande = ur7e_line_follower.analyse_commande:main',
            'component_diagnostics = ur7e_line_follower.component_diagnostics:main',
            'plot_results = ur7e_line_follower.plot_results:main',
            'control_smoke = ur7e_line_follower.control_smoke:main',
            'scene_initializer = ur7e_line_follower.scene_initializer:main',
            'full_diagnostic = ur7e_line_follower.full_diagnostic:main',
            'expert_follow = ur7e_line_follower.expert_follow:main',
            'phase1_repeatability = ur7e_line_follower.phase1_repeatability:main',
            'phase1_action_directions = ur7e_line_follower.phase1_action_directions:main',
            'visual_test   = ur7e_line_follower.visual_test:main',
            'training_monitor = ur7e_line_follower.training_monitor:main',
            'plot_training    = ur7e_line_follower.plot_training:main',
            'demo_gazebo      = ur7e_line_follower.demo_gazebo:main',
        ],
    },
)
