from setuptools import setup, find_packages

setup(
    name='seacliff',
    description="Data-driven simuilations of Rubin Observations",
    author="Matthew R Becker w/ LSST-DESC",
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
