from setuptools import setup

package_name = 'turtle_nn_control'

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
    description='Control basado en redes neuronales para TurtleSim usando PyTorch',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nn_controller = turtle_nn_control.turtle_nn_controller:main',
        ],
    },
)

