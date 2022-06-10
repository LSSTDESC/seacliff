import numpy as np

import lsst.geom
import lsst.afw.geom
import galsim
from galsim.wcs import CelestialWCS


class RubinSkyWCS(CelestialWCS):
    """A galsim-compatible wrapper of the Rubin SkyWcs class.

    NOTE: GalSim uses FITS 1-based indexing whereas Rubin uses 0-based indexing. This
    WCS class uses the galsim FITS conventions and explicitly translates via offsets of
    -1 or 1 as needed.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        The Rubin Sky WCS to be wrapped by this class.
    origin : PositionD or PositionI or None, optional
        If not None, the origin position of the image coordinate system. Note that
        the conversion from 1-based to 0-based pixel indexing is **always** done.
    """

    _req_params = {"wcs": lsst.afw.geom.SkyWcs}
    _opt_params = {"origin": galsim.PositionD}

    def __init__(self, wcs, origin=None):
        self._wcs = wcs
        self._wcs_str = wcs.writeString()
        self._set_origin(origin)
        # kept here so that functions in parent class can use it
        self._color = None

    @property
    def wcs(self):
        """the underlying lsst.afw.geom.SkyWcs object"""
        return self._wcs

    @property
    def wcs_str(self):
        """the string representation of the underlying lsst.afw.geom.SkyWcs object"""
        return self._wcs_str

    @property
    def origin(self):
        """origin in image coordinates"""
        return self._origin

    def _radec(self, x, y, color=None):
        if np.ndim(x) != np.ndim(y) or np.shape(x) != np.shape(y):
            raise RuntimeError(
                "x and y must have the same dimension and shape when converting to "
                "ra,dec in RubinSkyWCS! x dim/shape = %s/%s y dim/shape = %s/%s"
                % (np.dim(x), np.shape(x), np.dim(y), np.shape(y))
            )
        # the input x, y are in FITS conventions so we subtract 1 to get to
        # LSST conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        _x = np.atleast_1d(x) - 1
        _y = np.atleast_1d(y) - 1
        ra, dec = self.wcs.pixelToSkyArray(_x, _y, degrees=False)

        if np.ndim(x) == np.ndim(y) and np.ndim(y) == 0:
            return ra[0], dec[0]
        else:
            return ra, dec

    def _xy(self, ra, dec, color=None):
        if np.ndim(ra) != np.ndim(dec) or np.shape(ra) != np.shape(dec):
            raise RuntimeError(
                "ra and dec must have the same dimension and shape when converting "
                "to x,y in RubinSkyWCS! ra dim/shape = %s/%s dec dim/shape = "
                "%s/%s" % (np.dim(ra), np.shape(ra), np.dim(dec), np.shape(dec))
            )

        _ra = np.atleast_1d(ra)
        _dec = np.atleast_1d(dec)
        x, y = self.wcs.skyToPixelArray(_ra, _dec, degrees=False)

        # the output x, y are in Rubin conventions so we add 1 to get to
        # FITS conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        x += 1
        y += 1

        if np.ndim(ra) == np.ndim(dec) and np.ndim(ra) == 0:
            return x[0], y[0]
        else:
            return x, y

    def copy(self):
        """make a copy"""
        return self._newOrigin(self._origin)

    def _newOrigin(self, origin):
        return RubinSkyWCS(
            self.wcs.copyAtShiftedPixelOrigin(lsst.geom.Extent2D(0)),
            origin=origin,
        )

    def __eq__(self, other):
        return (self is other) or (
            isinstance(other, RubinSkyWCS)
            and
            # same as stack C++
            # xref: https://github.com/lsst/afw/blob/main/src/geom/SkyWcs.cc#L156
            self.wcs_str == other.wcs_str
            and self.origin == other.origin
        )

    def __repr__(self):
        return (
            "seacliff.RubinSkyWCS(lsst.afw.geom.SkyWcs.readString(%r), origin=%r)"
            % (
                self.wcs_str,
                self.origin,
            )
        )

    def __hash__(self):
        return hash(repr(self))

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["_wcs"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._wcs = lsst.afw.geom.SkyWcs.readString(self._wcs_str)
