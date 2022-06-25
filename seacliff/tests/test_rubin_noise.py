import numpy as np
from numpy.testing import assert_allclose

import galsim

from seacliff.rubin_noise import get_rubin_skyvar_and_gain
import lsst.afw.image


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
