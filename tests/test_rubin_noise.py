import numpy as np
import os
from numpy.testing import assert_allclose

import galsim

import pytest

from seacliff.testing import check_pickle_eval_repr_copy
from seacliff.rubin_noise import get_rubin_skyvar_and_gain, RubinNoise
import lsst.afw.image
import lsst.afw.math


def test_rubin_noise_fit_whole_image():
    rng = np.random.RandomState(seed=10)
    n = 101
    gain = 1.4

    gim = (
        galsim.Gaussian(fwhm=0.8).withFlux(10).drawImage(nx=n, ny=n, scale=0.2).array
        * gain
    )

    bkg = gim.max() * 100
    tot = gim + bkg
    gim_nse = rng.poisson(lam=tot)
    gim_nse = gim_nse - bkg

    exp = lsst.afw.image.ExposureF(n, n)
    exp.image.array[:, :] = gim_nse / gain
    exp.variance.array[:, :] = (gim_nse + bkg) / gain**2

    sv, gn = get_rubin_skyvar_and_gain(exp)
    assert_allclose(gn, gain)
    assert_allclose(sv, bkg / gain**2)


def test_rubin_noise_with_gains_dc2():
    exp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    exp_bkg = lsst.afw.math.BackgroundList.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp_bkg.fits")
    )
    bkg = exp_bkg.getImage().array
    gain = np.mean([amp.getGain() for amp in exp.getDetector().getAmplifiers()])

    sv, gn = get_rubin_skyvar_and_gain(exp)
    assert_allclose(gn, gain)
    # the bkg = sky counts but variance is sky counts / gain
    # we check for medians since some regions have huge counts due to artifacts,
    # saturation etc.
    assert_allclose(np.median(sv), np.median(bkg / gain), rtol=0.1, atol=0)

    exp.setDetector(None)
    sv, gn = get_rubin_skyvar_and_gain(exp)
    assert_allclose(gn, gain, atol=2e-2, rtol=0)


def test_rubin_noise_galsim():
    exp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    exp_bkg = lsst.afw.math.BackgroundList.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp_bkg.fits")
    )
    bkg = exp_bkg.getImage().array

    nse = RubinNoise(exp)
    gain = np.mean([amp.getGain() for amp in exp.getDetector().getAmplifiers()])
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    # the bkg = sky counts but variance is sky counts / gain
    # we check for medians since some regions have huge counts due to artifacts,
    # saturation etc.
    assert_allclose(
        np.median(nse.sky_level.array),
        np.median(bkg / gain),
        rtol=0.1,
        atol=0,
    )

    assert_allclose(
        nse.getVariance().array,
        nse.sky_level.array / nse.gain.array,
    )


def test_rubin_noise_pickle_eval_repr_copy():
    gain = galsim.ImageD(np.ones((10, 10)) * 0.7)
    sv = galsim.ImageD(np.ones((10, 10)) * 100)

    nse = RubinNoise(sv, gain=gain)
    check_pickle_eval_repr_copy(nse)

    # make sure this runs
    str(nse)


def test_rubin_noise_galsim_scale_variance():
    exp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    gain = np.mean([amp.getGain() for amp in exp.getDetector().getAmplifiers()])

    nse = RubinNoise(exp)
    new_var = np.median(nse.sky_level.array / nse.gain.array * 4)
    nse2 = nse.withVariance(new_var)
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    assert_allclose(nse2.getVariance().array, new_var)

    nse = RubinNoise(exp)
    nse2 = nse.withVariance(nse.sky_level / nse.gain * 4)
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    assert_allclose(
        nse2.getVariance().array,
        4 * nse.sky_level.array / nse.gain.array,
    )

    nse2 = nse.withScaledVariance(3)
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    assert_allclose(
        nse2.getVariance().array,
        3 * nse.sky_level.array / nse.gain.array,
    )

    nse = RubinNoise(exp)
    nse2 = nse * 4
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    assert_allclose(
        nse2.getVariance().array,
        4 * nse.sky_level.array / nse.gain.array,
    )

    nse2 = nse / 3
    assert_allclose(nse.gain.array, gain, atol=2e-2, rtol=0)
    assert_allclose(
        nse2.getVariance().array,
        nse.sky_level.array / nse.gain.array / 3,
    )

    nse = RubinNoise(10, gain=4)
    nse2 = nse.withVariance(nse.sky_level / nse.gain * 4)
    assert_allclose(nse.gain, 4, atol=2e-2, rtol=0)
    assert_allclose(
        nse2.getVariance(),
        4 * nse.sky_level / nse.gain,
    )


def test_rubin_noise_galsim_mad_clipping():
    exp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    nse = RubinNoise(exp)
    nse_clipped = RubinNoise(exp, mad_clipping=5)

    # we should have outliers
    assert np.abs(nse.sky_level.array).max() > np.abs(nse_clipped.sky_level.array).max()

    # a lot should be the same
    frac5 = np.mean(nse.sky_level.array == nse_clipped.sky_level.array)
    assert frac5 > 0.99
    assert frac5 < 1.00

    # more clipping means less the same
    nse_clipped = RubinNoise(exp, mad_clipping=1)
    frac1 = np.mean(nse.sky_level.array == nse_clipped.sky_level.array)
    assert frac1 < frac5


@pytest.mark.parametrize("sky_arr", [True, False])
@pytest.mark.parametrize("gain_arr", [True, False])
def test_rubin_noise_apply_image(gain_arr, sky_arr):
    x, y = np.meshgrid(np.arange(101), np.arange(101))
    if gain_arr:
        gain = galsim.ImageD(0.7 * (1 + y / 100 * 0.1))
    else:
        gain = 0.7

    if sky_arr:
        sv = galsim.ImageD(100 * (1 + x / 100 * 0.1))
    else:
        sv = 100

    nse = RubinNoise(sv, gain=gain)

    # do a basic check for image sizes
    if gain_arr or sky_arr:
        im = (
            galsim.Gaussian(fwhm=2.0).withFlux(1000).drawImage(scale=0.2, nx=10, ny=101)
        )
        with pytest.raises(RuntimeError):
            im.addNoise(nse)

    # do a basic check
    im = galsim.Gaussian(fwhm=2.0).withFlux(1000).drawImage(scale=0.2, nx=101, ny=101)
    im_orig = im.copy()
    im.addNoise(nse)
    assert np.any(im.array != im_orig.array)

    # make sure comes out different
    im2 = im_orig.copy()
    im2.addNoise(nse)
    assert np.any(im2.array != im.array)

    # check values of noise
    im_orig = (
        galsim.Gaussian(fwhm=2.0).withFlux(1000).drawImage(scale=0.2, nx=101, ny=101)
    )
    arrs = []
    for _ in range(10_000):
        im = im_orig.copy()
        im.addNoise(nse)
        arrs.append(im.array)

    mn = np.mean(arrs, axis=0)
    assert_allclose(mn, im_orig.array, atol=1, rtol=0)

    sd = np.std(arrs, axis=0)
    assert_allclose(sd, np.sqrt(((im_orig + sv) / gain).array), atol=0, rtol=0.1)

    # check seeding
    nse1 = RubinNoise(sv, gain=gain, rng=galsim.BaseDeviate(10))
    nse2 = RubinNoise(sv, gain=gain, rng=galsim.BaseDeviate(10))
    im_orig = (
        galsim.Gaussian(fwhm=2.0).withFlux(1000).drawImage(scale=0.2, nx=101, ny=101)
    )
    im1 = im_orig.copy()
    im1.addNoise(nse1)
    im2 = im_orig.copy()
    im2.addNoise(nse2)
    assert np.any(im1.array == im2.array)
