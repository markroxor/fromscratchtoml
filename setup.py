''' Installation script for python fromscratchtoml package'''

from setuptools import find_packages, setup

linux_testenv = [
    'pytest-cov',
    'python-coveralls',
]

setup(
    name='fromscratchtoml',
    version='0.0.1',
    description=' An intuitive machine learning library for python.',

    install_requires=[
        'numpy >= 1.14.2',
    ],
    extras_require={
        'test': linux_testenv,
        'docs': linux_testenv + ['sphinx', 'sphinxcontrib-napoleon', 'travis-sphinx', 'sphinxcontrib.programoutput']
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/jellAIfish/fromscratchtoml',
    author='Mohit Rathore',
    author_email='mrmohitrathoremr@gmail.com',
    license='GPL-3.0',
    packages=find_packages(),
)
