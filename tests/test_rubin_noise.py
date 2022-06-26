import numpy as np
import os
from numpy.testing import assert_allclose

import galsim

from seacliff.rubin_noise import get_rubin_skyvar_and_gain
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
