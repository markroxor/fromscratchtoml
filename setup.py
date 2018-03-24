''' Installation script for python omega package'''

import os
import sys
from setuptools import find_packages, setup

win_testenv = [
    'pytest',
],
linux_testenv = win_testenv

setup(
    name='omega',
    version='0.0.1',
    description=' An intuitive machine learning library for python.',

    install_requires=[
        'numpy >= 1.14.2',
    ],
    extras_require={
        'test-win': win_testenv,
        'test': linux_testenv,
    },
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/jellAIfish/omega',
    license='GPL-3.0',
    packages=find_packages(),
)
