import os
import tempfile

from numpy.testing import assert_allclose

import lsst.afw.image
import seacliff
import galsim

import pytest


@pytest.mark.parametrize("deconvolve_pixel", [True, False])
@pytest.mark.parametrize("use_calexp", [False])  # TODO add true in galsim 2.4
def test_config_psf(deconvolve_pixel, use_calexp):
    wcs_pth = os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    wcs = seacliff.RubinSkyWCS(
        lsst.afw.image.ExposureD.readFits(
            os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
        ).getWcs()
    )
    psf = seacliff.RubinPSF(
        lsst.afw.image.ExposureD.readFits(
            os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
        ).getPsf(),
        wcs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(str(tmpdir), "test.fits")
        config = {
            "modules": ["seacliff"],
            "input": {
                "calexp": {
                    "file_name": wcs_pth,
                },
            },
            "gal": {
                "type": "Exponential",
                "half_light_radius": 0.5,
                "flux": 1e5,
            },
            "psf": {
                "type": "RubinPSF",
                "psf": "$(@input.calexp).getPsf()",
                "deconvolve_pixel": deconvolve_pixel,
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

        if use_calexp:
            del config["image"]["wcs"]["wcs"]
            del config["psf"]["psf"]

        galsim.config.Process(config)
        assert os.path.exists(img_pth)
        img_pos = galsim.PositionD(27, 27)

        img = (
            galsim.Convolve(
                [
                    psf.getPSF(img_pos, deconvolve_pixel=deconvolve_pixel),
                    galsim.Exponential(half_light_radius=0.5, flux=1e5),
                ]
            )
            .drawImage(
                nx=53,
                ny=53,
                wcs=wcs.local(img_pos),
            )
            .array
        )

        img_config = galsim.fits.read(img_pth).array
        assert_allclose(img, img_config)


def test_config_psf_raises():
    wcs_pth = os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(str(tmpdir), "test.fits")

        # no PSF
        config = {
            "modules": ["seacliff"],
            "gal": {
                "type": "Exponential",
                "half_light_radius": 0.5,
                "flux": 1e5,
            },
            "psf": {
                "type": "RubinPSF",
                "deconvolve_pixel": False,
            },
            "image": {
                "type": "Single",
                "pixel_scale": 0.2,
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }
        with pytest.raises(RuntimeError) as e:
            galsim.config.Process(config)

        assert "RubinPSF" in str(e.value)

        # set properties wrong
        config = {
            "modules": ["seacliff"],
            "input": {
                "calexp": {
                    "file_name": wcs_pth,
                },
            },
            "gal": {
                "type": "Exponential",
                "half_light_radius": 0.5,
                "flux": 1e5,
            },
            "psf": {
                "type": "RubinPSF",
                "psf": "$(@input.calexp).getPsf()",
                "deconvolve_pixel": True,
                "depixelize": True,
            },
            "image": {
                "type": "Single",
                "pixel_scale": 0.2,
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }
        with pytest.raises(RuntimeError) as e:
            galsim.config.Process(config)

        assert "deconvolve_pixel" in str(e.value)
        assert "depixelize" in str(e.value)
