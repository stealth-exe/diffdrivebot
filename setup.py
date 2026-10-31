from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'diffdrivebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # includes scripts/ automatically
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        # Robot description (differential-drive bot)
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Mazzetto',
    maintainer_email='workabotic@gmail.com',
    description='An end-to-end deep learning pipeline for autonomous driving in ROS 2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'autopilot = diffdrivebot.autopilot:main',
            'data_collector = diffdrivebot.data_collector:main',
            'autopilot_rl = diffdrivebot.autopilot_rl:main',
            'autopilot_diffdrive = diffdrivebot.autopilot_diffdrive:main',
        ],
    },
)
