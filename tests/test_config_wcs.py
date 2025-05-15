import os
import tempfile

import galsim
import lsst.afw.image
import pytest
from numpy.testing import assert_allclose

import seacliff


@pytest.mark.parametrize("dtype", ["double", "float"])
def test_config_wcs(dtype):
    wcs_pth = os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    wcs = seacliff.RubinSkyWCS(
        lsst.afw.image.ExposureD.readFits(
            os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
        ).getWcs()
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(str(tmpdir), "test.fits")
        config = {
            "modules": ["seacliff"],
            "input": {
                "calexp": {
                    "file_name": wcs_pth,
                    "dtype": dtype,
                },
            },
            "gal": {
                "type": "Exponential",
                "half_light_radius": 0.5,
                "flux": 1e5,
            },
            "psf": {
                "type": "Gaussian",
                "fwhm": 0.8,
            },
            "image": {
                "type": "Single",
                "wcs": {
                    "type": "RubinSkyWCS",
                    "wcs": "$(@input.calexp).getWcs()",
                },
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }

        galsim.config.Process(config)
        assert os.path.exists(img_pth)

        img = (
            galsim.Convolve(
                [
                    galsim.Gaussian(fwhm=0.8),
                    galsim.Exponential(half_light_radius=0.5, flux=1e5),
                ]
            )
            .drawImage(
                nx=53,
                ny=53,
                wcs=wcs.local(galsim.PositionD(27, 27)),
            )
            .array
        )

        img_config = galsim.fits.read(img_pth).array
        assert_allclose(img, img_config)


def test_config_wcs_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(str(tmpdir), "test.fits")
        config = {
            "modules": ["seacliff"],
            "gal": {
                "type": "Exponential",
                "half_light_radius": 0.5,
                "flux": 1e5,
            },
            "psf": {
                "type": "Gaussian",
                "fwhm": 0.8,
            },
            "image": {
                "type": "Single",
                "wcs": {
                    "type": "RubinSkyWCS",
                },
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }

        with pytest.raises(RuntimeError) as e:
            galsim.config.Process(config)

        assert "RubinWCS" in str(e.value)
