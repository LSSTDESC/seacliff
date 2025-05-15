import galsim
import lsst.afw.geom
import lsst.geom
import numpy as np
from galsim.wcs import CelestialWCS


class RubinSkyWCS(CelestialWCS):
    """A galsim-compatible wrapper of the Rubin SkyWcs class.

    NOTE: GalSim uses FITS 1-based indexing whereas Rubin uses 0-based indexing. This
    WCS class uses the galsim FITS conventions and explicitly translates via offsets of
    -1 or 1 as needed.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        The Rubin Sky WCS to be wrapped by this class. Usually one can get this from the
        `.getWcs()` method attached to a calexp.
    """

    def __init__(self, wcs):
        self._wcs = wcs
        self._wcs_str = wcs.writeString()
        # we have the galsim origin mirror the Rubin one with the offset due to pixel
        # indexing convetions
        self._set_origin(
            galsim.PositionD(
                self.wcs.getPixelOrigin().x + 1,
                self.wcs.getPixelOrigin().y + 1,
            )
        )
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
                % (np.ndim(x), np.shape(x), np.ndim(y), np.shape(y))
            )
        # the input x, y are in FITS conventions so we subtract 1 to get to
        # LSST conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        # galsim subtracts the origin on the way in so we handle that here
        _x = np.atleast_1d(x) - 1 + self.origin.x
        _y = np.atleast_1d(y) - 1 + self.origin.y
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
                "%s/%s" % (np.ndim(ra), np.shape(ra), np.ndim(dec), np.shape(dec))
            )

        _ra = np.atleast_1d(ra)
        _dec = np.atleast_1d(dec)
        x, y = self.wcs.skyToPixelArray(_ra, _dec, degrees=False)

        # the output x, y are in Rubin conventions so we add 1 to get to
        # FITS conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        # galsim handles the origin itself so we account for that here
        # it will add the origin on the way out
        x += 1 - self.origin.x
        y += 1 - self.origin.y

        if np.ndim(ra) == np.ndim(dec) and np.ndim(ra) == 0:
            return x[0], y[0]
        else:
            return x, y

    def copy(self):
        """make a copy"""
        return self._newOrigin(self._origin)

    def _newOrigin(self, origin):
        return RubinSkyWCS(
            self.wcs.copyAtShiftedPixelOrigin(
                lsst.geom.Extent2D(
                    origin.x - self.origin.x,
                    origin.y - self.origin.y,
                ),
            ),
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
        return "seacliff.RubinSkyWCS(lsst.afw.geom.SkyWcs.readString(%r))" % (
            self.wcs_str,
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

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def _writeHeader(self, header, bounds):
        # this currently produces a TAN WCS (possibly approximate)
        # in the future it will be TAN-SIP: https://jira.lsstcorp.org/browse/DM-13170
        # the Rubin SkyWcs will add 1 on the way out here when making the header
        md = self.wcs.getFitsMetadata(False).toDict()
        for k, v in md.items():
            header[k] = v
