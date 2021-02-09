from setuptools import find_packages, setup
setup(
    name='aiga_train',
    packages=find_packages(include=['aiga_train']),
    entry_points = {
        'console_scripts': ['aiga_train=aiga_train.main:main'],
    },
    version='0.1.13',
    description='My first Python library',
    author='Me',
    license='MIT',
)