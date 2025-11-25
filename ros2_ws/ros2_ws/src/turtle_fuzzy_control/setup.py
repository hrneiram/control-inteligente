from setuptools import setup

package_name = 'turtle_fuzzy_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hanssel Neira',
    maintainer_email='ric-185@hotmail.com',
    description='Control difuso para TurtleSim para control inteligente',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Legacy entry retained for historical reference only:
            # 'fuzzy_controller = turtle_fuzzy_control.turtle_fuzzy_controller:main',
            'fuzzy_visual = turtle_fuzzy_control.turtle_fuzzy_obstacles_visual:main',
        ],
    },
)
