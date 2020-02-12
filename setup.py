"""Install robovat.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


with open('README.md', 'r') as fh:
    long_description = fh.read()


setup(
    name='robovat',
    version='0.1.0',
    author='Kuan Fang',
    author_email='kuanfang@ai.stanford.edu',
    license='MIT',
    packages=find_packages(),
    description=('RoboVat: A unified toolkit for simulated and real-world '
                 'robotic task environments.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StanfordVL/robovat',
    install_requires=[
        'easydict==1.9',
        'future==0.17.1',
        'gym==0.14.0',
        'h5py==2.9.0',
        'matplotlib==2.2.4',
        'numpy==1.16.5',
        'opencv-python==4.1.0.25',
        'Pillow==6.1.0',
        'python-pcl',
        'PyYAML==5.1.2',
        'scikit-learn==0.20.4',
        'scipy==1.2.2',
        'six==1.11.0',
        'sklearn==0.0',
        'pybullet==2.6.5',
    ],
    include_package_data=False,
    zip_safe=False,
)
