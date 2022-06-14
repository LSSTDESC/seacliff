# this bloack ensures this code works for python < 3.8 where importlib.metadata is
# not in the stdlib
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("seacliff")
except PackageNotFoundError:
    # package is not installed
    pass

from .rubin_sky_wcs import RubinSkyWCS  # noqa
