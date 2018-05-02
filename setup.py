''' Installation script for python fromscratchtoml package'''

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

linux_testenv = [
    'pytest-cov',
    'python-coveralls',
]

setup(
    name='fromscratchtoml',
    version='0.0.2',
    description=' An intuitive machine learning library for python.',

    install_requires=install_requires,
    extras_require={
        'test': linux_testenv,
        'docs': ['jupyter', 'Flask>=0.10.1', 'Jinja2>=2.7', 'MarkupSafe>=0.18', 'Werkzeug>=0.9.1',
'               itsdangerous>=0.22', 'flask-flatpages', 'frozen-flask', 'flask-assets']
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3.0',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/jellAIfish/fromscratchtoml',
    author='markroxor',
    author_email='mrmohitrathoremr@gmail.com',
    license='GPL-3.0',
    packages=find_packages(),
)
