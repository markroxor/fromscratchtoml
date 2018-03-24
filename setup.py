''' Installation script for python wAlnut package'''

import os
import sys
from setuptools import find_packages, setup

setup(
    name='omega',
    version='0.0.1',
    description=' An intuitive machine learning library for python.',
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/jellAIfish/omega',
    license='GPL-3.0',
    # Automatically find packages inside wAlnut to install
    packages=find_packages(),
)
