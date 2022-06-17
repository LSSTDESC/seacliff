import os
import copy

import pytest

import galsim
import seacliff
import lsst.afw.detection
import lsst.geom
import galsim.hsm

from numpy.testing import assert_allclose

from seacliff.testing import check_pickle_eval_repr_copy


def test_rubin_psf_init():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    seacliff.RubinPSF(pth, galsim.PixelScale(0.2))

    with open(pth, "rb") as fp:
        psf_bytes = fp.read()

    seacliff.RubinPSF(psf_bytes, galsim.PixelScale(0.2))

    psf = lsst.afw.detection.Psf.readFits(pth)
    gpsf = seacliff.RubinPSF(psf, galsim.PixelScale(0.2))
    assert gpsf.psf is psf

    with pytest.raises(ValueError):
        seacliff.RubinPSF(psf, None)


def test_rubin_psf_pickle_eval_repr_copy():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    psf = seacliff.RubinPSF(pth, galsim.PixelScale(0.2))

    check_pickle_eval_repr_copy(psf)


def test_rubin_psf_equal():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    psf = lsst.afw.detection.Psf.readFits(pth)
    gpsf1 = seacliff.RubinPSF(psf, galsim.PixelScale(0.2))
    assert gpsf1 == gpsf1

    gpsf2 = seacliff.RubinPSF(psf, galsim.PixelScale(0.22))
    assert gpsf1 != gpsf2
    assert gpsf1.psf is gpsf2.psf

    gpsf3 = copy.copy(gpsf1)
    assert gpsf1 == gpsf3
    assert gpsf1.psf is not gpsf3.psf
    assert gpsf1.wcs is not gpsf3.wcs

    gpsf4 = copy.deepcopy(gpsf1)
    assert gpsf1 == gpsf4
    assert gpsf1.psf is not gpsf4.psf
    assert gpsf1.wcs is not gpsf4.wcs


def test_rubin_psf_correct():
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    rpsf = lsst.afw.detection.Psf.readFits(pth)
    gpsf = seacliff.RubinPSF(rpsf, galsim.PixelScale(0.2))

    # off center PSF
    x = 450.3
    y = 451.2
    image_pos = galsim.PositionD(x, y)
    rubin_pos = lsst.geom.Point2D(x - 1, y - 1)
    gpsf_im = gpsf.getPSF(image_pos)
    rpsf_im = rpsf.computeImage(rubin_pos)

    # check that underlying data is correct
    assert_allclose(gpsf_im.image.array, rpsf_im.array / rpsf_im.array.sum())

    # check that we can draw it back
    dx = x - int(x + 0.5)
    dy = y - int(y + 0.5)
    dpsf = gpsf_im.drawImage(
        nx=rpsf_im.array.shape[0],
        ny=rpsf_im.array.shape[1],
        scale=0.2,
        offset=(dx, dy),
        method="no_pixel",
    )
    assert_allclose(dpsf.array, rpsf_im.array / rpsf_im.array.sum(), rtol=5e-7)

    # now let's try a pixel center
    x = 450
    y = 471
    image_pos = galsim.PositionD(x, y)
    rubin_pos = lsst.geom.Point2D(x - 1, y - 1)
    gpsf_im = gpsf.getPSF(image_pos)
    rpsf_im = rpsf.computeImage(rubin_pos)

    # same checks as above
    assert_allclose(gpsf_im.image.array, rpsf_im.array / rpsf_im.array.sum())
    dpsf = gpsf_im.drawImage(
        nx=rpsf_im.array.shape[0],
        ny=rpsf_im.array.shape[1],
        scale=0.2,
        method="no_pixel",
    )
    assert_allclose(dpsf.array, rpsf_im.array / rpsf_im.array.sum(), rtol=5e-7)

    # shift by pixels
    dpsf = gpsf_im.drawImage(
        nx=rpsf_im.array.shape[0],
        ny=rpsf_im.array.shape[1],
        scale=0.2,
        method="no_pixel",
        offset=(1, 3),
    )
    assert_allclose(
        dpsf.array[3:, 1:],
        rpsf_im.array[:-3, :-1] / rpsf_im.array.sum(),
        rtol=5e-7,
    )


def test_rubin_psf_color_does_nothing():
    # As of writing (2022-06-17), the Rubin PSF model does not have color. Thus
    # this keyword arg should do nothing. -- MRB
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    rpsf = lsst.afw.detection.Psf.readFits(pth)
    gpsf = seacliff.RubinPSF(rpsf, galsim.PixelScale(0.2))

    x = 450
    y = 471
    image_pos = galsim.PositionD(x, y)
    psf1 = gpsf.getPSF(image_pos)
    psf2 = gpsf.getPSF(image_pos, color=10)

    assert psf1 == psf2


def test_rubin_psf_center():
    # As of writing (2022-06-17), the Rubin PSF model does not have color. Thus
    # this keyword arg should do nothing. -- MRB
    pth = os.path.join(os.path.dirname(__file__), "data", "rubin_psf.fits")
    rpsf = lsst.afw.detection.Psf.readFits(pth)
    gpsf = seacliff.RubinPSF(rpsf, galsim.PixelScale(0.2))

    # pixel center everywhere
    x = 450
    y = 471
    image_pos = galsim.PositionD(x, y)
    psf1 = gpsf.getPSF(image_pos)
    psf_im = psf1.drawImage(nx=53, ny=43, scale=0.2)
    mom = galsim.hsm.FindAdaptiveMom(psf_im)
    assert_allclose(mom.moments_centroid.x, 27, atol=5e-3)
    assert_allclose(mom.moments_centroid.y, 22, atol=5e-3)

    # slightly off the pixel center
    x = 450.1
    y = 471.3
    image_pos = galsim.PositionD(x, y)
    psf1 = gpsf.getPSF(image_pos)
    psf_im = psf1.drawImage(nx=53, ny=43, scale=0.2, offset=(0.1, 0.3))
    mom = galsim.hsm.FindAdaptiveMom(psf_im)
    assert_allclose(mom.moments_centroid.x, 27.1, atol=5e-3)
    assert_allclose(mom.moments_centroid.y, 22.3, atol=5e-3)

    # make sure we recenter ok
    x = 450.1
    y = 471.3
    image_pos = galsim.PositionD(x, y)
    psf1 = gpsf.getPSF(image_pos)
    psf_im = psf1.drawImage(nx=53, ny=43, scale=0.2)
    mom = galsim.hsm.FindAdaptiveMom(psf_im)
    assert_allclose(mom.moments_centroid.x, 27, atol=5e-3)
    assert_allclose(mom.moments_centroid.y, 22, atol=5e-3)
