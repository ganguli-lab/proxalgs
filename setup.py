import os
from distutils.core import setup

version = '0.2.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

setup(name='proxalgs',
      version=version,
      description='Proximal algorithms in python',
      long_description=README,
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/ganguli-lab/proxalgs',
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=['proxalgs'],
      requires=install_requires,
      extras_require={
          'dev': ['sphinx', 'sphinx-rtd-theme'],
          'test': ['nose']
      },
)
