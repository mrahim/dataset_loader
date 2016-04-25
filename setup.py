#! /usr/bin/env python
descr = """Dataset Loader"""

import sys
import os

from setuptools import setup, find_packages


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DISTNAME = 'dataset_loader'
DESCRIPTION = 'Dataset Loader'
with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Mehdi Rahim'
MAINTAINER_EMAIL = 'rahim.mehdi@gmail.com'
URL = 'None'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'None'
VERSION = "0.0"


if __name__ == "__main__":

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
          ],
          packages=find_packages())
