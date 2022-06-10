import numpy as np
from numpy.testing import assert_allclose

import galsim
import lsst.geom


import seacliff


def _make_wcs(seed):
    rng = np.random.RandomState(seed=seed)
    crpix = lsst.geom.Point2D(
        rng.randint(low=100, high=200),
        rng.randint(low=100, high=200),
    )
    scale = rng.uniform(low=0.18, high=0.22)
    theta = rng.uniform(low=0, high=2.0*np.pi)
    cd_matrix = np.array([
        [scale, 0.0],
        [0.0, scale]],
    )
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    rotmat = np.array([
        [costheta, -sintheta],
        [sintheta, costheta]],
    )
    cd_matrix = np.dot(cd_matrix, rotmat)
    ra = rng.uniform(low=0, high=2.0*np.pi)
    dec = np.arcsin(rng.uniform(-1, 1))
    crval = lsst.geom.SpherePoint(ra, dec, lsst.geom.radians)
    return lsst.afw.geom.makeSkyWcs(
        crpix=crpix,
        crval=crval,
        cdMatrix=cd_matrix,
    )


def _test_pos(wcs1, wcs2):
    ra1, dec1 = wcs1.xyToradec(
        np.array([10, 9]),
        np.array([78, 55]),
        units=galsim.degrees,
    )
    ra2, dec2 = wcs2.xyToradec(
        np.array([10, 9]),
        np.array([78, 55]),
        units=galsim.degrees,
    )
    assert_allclose(ra1, ra2)
    assert_allclose(dec1, dec2)

    x1, y1 = wcs1.xyToradec(ra1, dec1, units=galsim.degrees)
    x2, y2 = wcs2.xyToradec(ra2, dec2, units=galsim.degrees)
    assert_allclose(x1, x2)
    assert_allclose(y1, y2)


def test_rubin_sky_wcs_equal():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    assert wcs1 == wcs1

    wcs2 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    assert wcs1 == wcs2

    wcs3 = seacliff.RubinSkyWCS(_make_wcs(10))
    assert wcs3 == wcs3
    assert wcs1 != wcs3
    assert wcs2 != wcs3

    wcs4 = seacliff.RubinSkyWCS(_make_wcs(1))
    assert wcs1 != wcs4
    assert wcs2 != wcs4
    assert wcs3 != wcs4
    assert wcs4 == wcs4


def test_rubin_sky_wcs_eval_repr():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    wcs5 = eval(repr(wcs1))
    assert wcs5 == wcs1
    assert wcs5 is not wcs1

    _test_pos(wcs1, wcs5)


def test_rubin_sky_wcs_copy():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    wcs6 = wcs1.copy()
    assert wcs6 == wcs1
    assert wcs6 is not wcs1

    _test_pos(wcs1, wcs6)
