"""
Create docknet as a Python package
"""

from __future__ import print_function
import io
import os
import os.path
import sys
from typing import List, Tuple

from setuptools import find_packages, setup


MIN_PYTHON_VERSION = (3, 9)

PKGNAME = 'docknet'
DESC = '''
A pure NumPy implementation of neural networks
'''

# --------------------------------------------------------------------


def pkg_version():
    """Read the package version from VERSION.txt"""
    basedir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(basedir, 'VERSION.txt'), 'r') as f:
        return f.readline().strip()


def parse_requirements(filename: str = 'requirements.txt'
                       ) -> Tuple[List[str], List[str]]:
    """Read the requirements file"""
    pathname = os.path.join(os.path.dirname(__file__), filename)
    modules = []
    urls = []
    with io.open(pathname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                urls.append(line)
            else:
                modules.append(line)
    return modules, urls


# --------------------------------------------------------------------

VERSION = pkg_version()

if sys.version_info < MIN_PYTHON_VERSION:
    sys.exit(
        f'**** Sorry, {PKGNAME} {VERSION} needs at least Python '
        f'{".".join(map(str, MIN_PYTHON_VERSION))}'
    )

install_requires, dependency_links = parse_requirements()

setup_args = dict(
    # Metadata
    name=PKGNAME,
    version=VERSION,
    description=DESC.split('\n')[0],
    long_description=DESC,
    license='(c) Accenture',
    author='Javier Sastre',
    author_email='j.sastre.martinez@accenture.com',

    # Locate packages
    packages=find_packages(exclude=['tests']),

    # Requirements
    python_requires='>='+'.'.join(map(str, MIN_PYTHON_VERSION)),
    install_requires=install_requires,
    dependency_links=dependency_links,
    # Optional requirements
    extras_require={
        'test': ['pytest', 'nose', 'coverage'],
    },

    entry_points={'console_scripts': [
        'docknet_generate_data = docknet.generate_data:main',
        'docknet_evaluate = docknet.evaluate:main',
        'docknet_predict = docknet.predict:main',
        'docknet_start = docknet.app:main',
        'docknet_train = docknet.train:main'
    ]},

    include_package_data=False,  # otherwise, package_data is not used
    package_data={
        PKGNAME: [
            'resources/config.yaml',
            'resources/chessboard.pkl',
            'resources/cluster.pkl',
            'resources/island.pkl',
            'resources/swirl.pkl'
        ]
    },

    # pytest requirements
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # More metadata
    keywords=['Accenture', 'The Dock', 'deep learning', 'neural network',
              'numpy', 'docknet'],
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'License :: Other/Proprietary License',
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],
)


if __name__ == '__main__':
    setup(**setup_args)
