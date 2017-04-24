from setuptools import setup, find_packages
import proxalgs

setup(
    name='proxalgs',
    version=proxalgs.__version__,
    description='Proximal algorithms in python',
    long_description='Proximal algorithms for convex optimization.',
    author='Niru Maheshwaranathan',
    author_email='nirum@stanford.edu',
    url='https://github.com/ganguli-lab/proxalgs',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'toolz', 'tableprint'],
    extras_require={
        'dev': ['sphinx', 'sphinx-rtd-theme'],
        'test': ['nose']
    },
)
