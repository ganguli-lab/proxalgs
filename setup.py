from distutils.core import setup

setup(name='proxalgs',
      version='0.1',
      description='Proximal algorithms in python',
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/ganguli-lab/proxalgs.git',
      requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
            Contains tools for running proximal algorithms to minimize an
            objective consisting of a sum of multiple functions. Exposes
            a bunch of useful proximal operators for incorporating common
            regularization terms in certain proximal algorithms.
            ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=['proxalgs'],
      package_dir={'proxalgs': ''},
      py_modules=['proxalgs,operators'],
      license='LICENSE.md'
)
