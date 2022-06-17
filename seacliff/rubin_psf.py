import copy
import galsim

import lsst.geom
import lsst.afw.image
from lsst.afw.fits import MemFileManager
import lsst.afw.detection


class RubinPSF(object):
    """A galsim-compatible wrapper for Rubin PSF models.

    NOTE: GalSim uses FITS 1-based indexing whereas Rubin uses 0-based indexing. This
    WCS class uses the galsim FITS conventions and explicitly translates via offsets of
    -1 or 1 as needed.

    Parameters
    ----------
    psf : lsst.afw.detection.Psf, a subclass thereof, bytes, or string
        The Rubin PSF model. Usually one can get this from the `.getPSF()` method
        attached to a calexp. You can also pass the raw bytes containing the
        serialized rubin PSF model. You can finally pass along a path to a FITS file
        with the PSF.
    wcs : galsim.BaseWCS or a subclass thereof
        The WCS class used to map the PSF pixels to world coordinates.
    """

    def __init__(self, psf, wcs):
        if isinstance(psf, str):
            self._psf = lsst.afw.detection.Psf.readFits(psf)
        elif isinstance(psf, bytes):
            mem = MemFileManager(len(psf))
            mem.setData(psf, len(psf))
            self._psf = lsst.afw.detection.Psf.readFits(mem)
            self._psf_bytes = psf
        else:
            self._psf = psf

        # doing this just in case someone decides to pass in a WCS from Rubin here
        if not isinstance(wcs, galsim.BaseWCS):
            raise RuntimeError("The WCS class passed to RubinPSF must be from galsim!")

        self._wcs = wcs

    @property
    def wcs(self):
        """the underlying galsim WCS object"""
        return self._wcs

    @property
    def psf(self):
        """the underlying Rubin PSF object"""
        return self._psf

    @property
    def psf_bytes(self):
        """the bytes of the FITS file representing this PSF"""
        if not hasattr(self, "_psf_bytes"):
            mem = MemFileManager()
            self._psf.writeFits(mem)
            self._psf_bytes = mem.getData()

        return self._psf_bytes

    def getPSF(self, image_pos, color=None, gsparams=None, **kwargs):
        """Get the PSF at a position in the image.

        Parameters
        ----------
        image_pos : galsim.PositionD
            The position in the image at which to draw the PSF. This position should be
            in 1-offset FITS coordinates, not 0-offset Rubin coordinates.
        color : float or None, optional
            The color for the PSF model. Rubin's internal code comments indicate
            that the meaning and structure of this argument is subject to change.
        gsparams : galsim.GSParams or None, optional
            An optional galsim parameter class for controlling the returned
            InterpolateImage.
        **kwargs : extra keywords, optional
            All extra keyword arguments are passed to the galsim.InterpolateImage
            returned by this method. The `offset`, `gsparams`, `wcs`, and
            `use_true_center` keywords are set internally. Passing these will cause
            an error to be raised.

        Returns
        -------
        im : galsim.InterpolateImage
            An InterpolateImage with the PSF profile.
        """
        rubin_pos = lsst.geom.Point2D(image_pos.x - 1, image_pos.y - 1)
        if color is not None:
            rubin_color = lsst.afw.image.Color(color)
            rubin_im = self.psf.computeImage(rubin_pos, color=rubin_color)
        else:
            rubin_im = self.psf.computeImage(rubin_pos)

        jac = self.wcs.jacobian(image_pos=image_pos)

        # this math is all in zero-offset pixel centered coordinates
        # it gets the offset of the PSF center to what
        # galsim will assume is the center of the profile
        start_x = rubin_im.getBBox().beginX
        start_y = rubin_im.getBBox().beginY
        dim_x = rubin_im.getDimensions().x
        dim_y = rubin_im.getDimensions().y
        cen_x = (dim_x - 1) / 2 + start_x
        cen_y = (dim_y - 1) / 2 + start_y
        offset_x = rubin_pos.x - cen_x
        offset_y = rubin_pos.y - cen_y

        # python will throw an error if the user duplicates any of the keywords in
        # kwargs - that's what we want since we want to set some of these ourselves
        im = galsim.InterpolateImage(
            galsim.ImageD(rubin_im.array / rubin_im.array.sum()),
            wcs=jac,
            offset=(offset_x, offset_y),
            gsparams=gsparams,
            use_true_center=True,  # we need this for the math above for the offset
            **kwargs,
        )
        return im

    def copy(self, memo=None):
        """make a copy"""
        # the `clone` method causes madness so make a copy the slow way
        mem = MemFileManager(len(self.psf_bytes))
        mem.setData(self.psf_bytes, len(self.psf_bytes))
        psf = lsst.afw.detection.Psf.readFits(mem)
        if memo is not None:
            return RubinPSF(psf, copy.deepcopy(self.wcs, memo))
        else:
            return RubinPSF(psf, copy.deepcopy(self.wcs))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(memo=memo)

    def __eq__(self, other):
        return (self is other) or (
            isinstance(other, RubinPSF)
            and
            # test if their fits representations are the same
            self.psf_bytes == other.psf_bytes
            and self.wcs == other.wcs
        )

    def __repr__(self):
        return "seacliff.RubinPSF(%r, %r)" % (
            self.psf_bytes,
            self.wcs,
        )

    def __hash__(self):
        return hash(repr(self))

    def __getstate__(self):
        # make sure we have the bytes attached
        self.psf_bytes

        d = self.__dict__.copy()
        del d["_psf"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        mem = MemFileManager(len(self._psf_bytes))
        mem.setData(self._psf_bytes, len(self._psf_bytes))
        self._psf = lsst.afw.detection.Psf.readFits(mem)
