import numpy as np

import lsst.afw.geom
import galsim
from galsim.wcs import CelestialWCS


class RubinSkyWCS(CelestialWCS):
    """A galsim-compatible wrapper of the Rubin SkyWCS class.

    NOTE: GalSim uses FITS 1-based indexing whereas Rubin uses 0-based indexing. This
    WCS class uses the galsim FITS conventions and explicitly translates via offsets of
    -1 or 1 as needed.

    Parameters
    ----------
    rubin_sky_wcs : lsst.afw.geom.SkyWCS
        The Rubin Sky WCS to be wrapped by this class.
    origin : PositionD or PositionI or None, optional
        If not None, the origin position of the image coordinate system. Note that
        the conversion from 1-based to 0-based pixel indexing is **always** done.
    """
    _req_params = {"rubin_sky_wcs": lsst.afw.geom.SkyWCS}
    _opt_params = {"origin": galsim.PositionD}

    def __init__(self, rubin_sky_wcs, origin=None):
        self._rubin_sky_wcs = rubin_sky_wcs
        self._set_origin(origin)
        # kept here so that functions in parent class can use it
        self._color = None

    @property
    def wcs(self):
        """the underlying lsst.afw.geom.SkyWCS object"""
        return self._rubin_sky_wcs

    @property
    def origin(self):
        """origin in image coordinates"""
        return self._origin

    def _radec(self, x, y, color=None):
        if np.ndim(x) != np.ndim(y) or np.shape(x) != np.shape(y):
            raise RuntimeError(
                "x and y must have the same dimension and shape when converting to "
                "ra,dec in RubinSkyWCS! x dim/shape = %s/%s y dim/shape = %s/%s" % (
                    np.dim(x), np.shape(x), np.dim(y), np.shape(y)
                )
            )
        # the input x, y are in FITS conventions so we subtract 1 to get to
        # LSST conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        _x = np.atleast_1d(x) - 1
        _y = np.atleast_1d(y) - 1
        ra, dec = self._rubin_sky_wcs.pixelToSkyArray(_x, _y, degree=False)

        if np.ndim(x) == np.ndim(y) and np.ndim(y) == 0:
            return ra[0], dec[0]
        else:
            return ra, dec

    def _xy(self, ra, dec, color=None):
        if np.ndim(ra) != np.ndim(dec) or np.shape(ra) != np.shape(dec):
            raise RuntimeError(
                "ra and dec must have the same dimension and shape when converting "
                "to x,y in RubinSkyWCS! ra dim/shape = %s/%s dec dim/shape = "
                "%s/%s" % (
                    np.dim(ra), np.shape(ra), np.dim(dec), np.shape(dec)
                )
            )

        _ra = np.atleast_1d(ra)
        _dec = np.atleast_1d(dec)
        x, y = self._rubin_sky_wcs.skyToPixelArray(_ra, _dec, degrees=False)

        # the output x, y are in Rubin conventions so we add 1 to get to
        # FITS conventions
        # see https://github.com/lsst/afw/blob/main/include/lsst/afw/geom/SkyWcs.h#L92
        x += 1
        y += 1

        if np.ndim(ra) == np.ndim(dec) and np.ndim(ra) == 0:
            return x[0], y[0]
        else:
            return x, y

    def _newOrigin(self, origin):
        # TODO - figure out how to make a copy?
        return RubinSkyWCS(self._rubin_sky_wcs, origin=origin)

    def __eq__(self, other):
        return (
            (self is other)
            or
            (
                isinstance(other, RubinSkyWCS)
                and
                # TODO figure out == for SkyWCS
                self._rubin_sky_wcs is other._rubin_sky_wcs
                and
                self.origin == other.origin
            )
        )

    # TODO add this and eval-able?
    # def __repr__(self):
    #     # rely on the repr from Rubin, w/e that is?
    #     return "seacliff.RubinSkyWCS(%r, origin=%r)" % (
    #         self._rubin_sky_wcs,
    #         self.origin,
    #     )
    #
    # def __hash__(self):
    #     return hash(repr(self))
