import numpy as np
from numpy.testing import assert_allclose
import pytest

import galsim
import lsst.geom
import seacliff
from seacliff.testing import check_pickle_eval_repr_copy


XTEST = np.array([10.0, 9.0])
YTEST = np.array([78.0, 55.0])


def _make_wcs(seed):
    rng = np.random.RandomState(seed=seed)
    crpix = lsst.geom.Point2D(
        rng.randint(low=100, high=200),
        rng.randint(low=100, high=200),
    )
    scale = rng.uniform(low=0.18, high=0.22)
    theta = rng.uniform(low=0, high=2.0 * np.pi)
    cd_matrix = np.array(
        [[scale, 0.0], [0.0, scale]],
    )
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    rotmat = np.array(
        [[costheta, -sintheta], [sintheta, costheta]],
    )
    cd_matrix = np.dot(cd_matrix, rotmat)
    ra = rng.uniform(low=0, high=2.0 * np.pi)
    dec = np.arcsin(rng.uniform(-1, 1))
    crval = lsst.geom.SpherePoint(ra, dec, lsst.geom.radians)
    return lsst.afw.geom.makeSkyWcs(
        crpix=crpix,
        crval=crval,
        cdMatrix=cd_matrix,
    )


def _test_pos(wcs1, wcs2, shift=None):
    if shift is not None:
        dx = shift.x
        dy = shift.y
    else:
        dx = 0
        dy = 0
    ra1, dec1 = wcs1.xyToradec(
        XTEST,
        YTEST,
        units=galsim.degrees,
    )
    ra2, dec2 = wcs2.xyToradec(
        XTEST + dx,
        YTEST + dy,
        units=galsim.degrees,
    )
    assert_allclose(ra1, ra2)
    assert_allclose(dec1, dec2)

    x1, y1 = wcs1.radecToxy(ra1, dec1, units=galsim.degrees)
    x2, y2 = wcs2.radecToxy(ra2, dec2, units=galsim.degrees)
    assert_allclose(x1, x2 - dx)
    assert_allclose(y1, y2 - dy)

    assert_allclose(x1, XTEST)
    assert_allclose(y1, YTEST)


@pytest.mark.parametrize("x,y", [(XTEST, YTEST), (XTEST[0], YTEST[0])])
def test_rubin_sky_wcs_correct(x, y):
    x0 = 10
    y0 = 2

    wcs = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(x0, y0))
    ra, dec = wcs.xyToradec(x, y, units=galsim.degrees)

    ra_rubin, dec_rubin = wcs.wcs.pixelToSkyArray(
        np.atleast_1d(x - x0 - 1),
        np.atleast_1d(y - y0 - 1),
        degrees=True,
    )
    assert_allclose(ra, ra_rubin)
    assert_allclose(dec, dec_rubin)

    x1, y1 = wcs.radecToxy(ra, dec, units=galsim.degrees)
    assert_allclose(x, x1)
    assert_allclose(y, y1)


def test_rubin_sky_wcs_raises():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    with pytest.raises(RuntimeError):
        wcs1.xyToradec(np.array([10]), np.array([10, 10]), units=galsim.degrees)

    with pytest.raises(RuntimeError):
        wcs1.radecToxy(np.array([10]), np.array([10, 10]), units=galsim.degrees)


def test_rubin_sky_wcs_equal():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    assert wcs1 == wcs1
    _test_pos(wcs1, wcs1)

    wcs2 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    assert wcs1 == wcs2
    assert wcs2 is not wcs1
    _test_pos(wcs1, wcs2)

    wcs3 = seacliff.RubinSkyWCS(_make_wcs(10))
    assert wcs3 == wcs3
    assert wcs1 != wcs3
    assert wcs2 != wcs3
    assert wcs3 is not wcs1

    wcs4 = seacliff.RubinSkyWCS(_make_wcs(1))
    assert wcs1 != wcs4
    assert wcs2 != wcs4
    assert wcs3 != wcs4
    assert wcs4 == wcs4
    assert wcs4 is not wcs1


def test_rubin_sky_wcs_pickle_eval_repr_copy():
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    wcs6 = wcs1.copy()
    assert wcs6 == wcs1
    assert wcs6 is not wcs1
    assert wcs1.wcs is not wcs6.wcs

    _test_pos(wcs1, wcs6)
    check_pickle_eval_repr_copy(wcs1, extra_tests=_test_pos)


def test_rubin_sky_wcs_origin():
    dx = 3
    dy = -2
    wcs1 = seacliff.RubinSkyWCS(_make_wcs(10), origin=galsim.PositionD(10, 2))
    wcs2 = wcs1.shiftOrigin(galsim.PositionD(dx, dy))
    _test_pos(wcs1, wcs2, shift=galsim.PositionD(dx, dy))

    assert wcs1.wcs is not wcs2.wcs
