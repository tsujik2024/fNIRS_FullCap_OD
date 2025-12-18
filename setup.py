from setuptools import setup

setup(
    name='fNIRS_FullCap_2025',
    version='',
    packages=['fnirs_FullCap_2025', 'fnirs_FullCap_2025.cli', 'fnirs_FullCap_2025.viz', 'fnirs_FullCap_2025.read',
              'fnirs_FullCap_2025.processing', 'fnirs_FullCap_2025.preprocessing'],
    url='',
    license='',
    author='Keiko Tsuji',
    author_email='tsujik@ohsu.edu',
    description='A Python pipeline for preprocessing and analyzing full cap fNIRS data (Artinis Medical Systems).',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'openpyxl',     # for Excel support
        'tqdm',
        'natsort',
        'setuptools']
)
