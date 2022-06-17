import os

import galsim
import seacliff
import lsst.afw.detection

from seacliff.testing import check_pickle_eval_repr_copy


def test_rubin_psf_io():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    seacliff.RubinPSF(pth, galsim.PixelScale(0.2))

    with open(pth, "rb") as fp:
        psf_bytes = fp.read()

    seacliff.RubinPSF(psf_bytes, galsim.PixelScale(0.2))

    psf = lsst.afw.detection.Psf.readFits(pth)
    seacliff.RubinPSF(psf, galsim.PixelScale(0.2))


def test_rubin_psf_pickle_eval_repr_copy():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    psf = seacliff.RubinPSF(pth, galsim.PixelScale(0.2))

    check_pickle_eval_repr_copy(psf)
