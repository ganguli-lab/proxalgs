import os
from distutils.core import setup

version = '0.2.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = ['numpy', 'scipy', 'hyperopt', 'sktensor']

tests_require = ['nose']

docs_require = ['Sphinx']

setup(name='proxalgs',
      version=version,
      description='Proximal algorithms in python',
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/ganguli-lab/proxalgs',
      requires=install_requires,
      license='MIT',
      long_description=README,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=['proxalgs']
)