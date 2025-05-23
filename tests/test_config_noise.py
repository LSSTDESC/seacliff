import copy
import os
import tempfile

import galsim
import lsst.afw.image
import numpy as np
import pytest
from numpy.testing import assert_allclose

import seacliff
from seacliff.config.noise import RubinNoiseBuilder


@pytest.mark.parametrize("include_obj_var", [True, False])
@pytest.mark.parametrize(
    "draw_method,flux",
    [
        ("phot", 0),
        ("phot", 1e5),
        ("fft", 0),
        ("fft", 1e10),
    ],
)
def test_config_noise(draw_method, flux, include_obj_var):
    wcs_pth = os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    calexp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    wcs = seacliff.RubinSkyWCS(calexp.getWcs())
    nse = seacliff.RubinNoise(calexp)
    nse_build = RubinNoiseBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(str(tmpdir), "test.fits")
        wgt_pth = os.path.join(str(tmpdir), "wgt.fits")
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
                "flux": flux,
            },
            "psf": {
                "type": "Gaussian",
                "fwhm": 0.8,
            },
            "stamp": {
                "draw_method": draw_method,
            },
            "image": {
                "type": "Single",
                "wcs": {
                    "type": "RubinSkyWCS",
                    "wcs": "$(@input.calexp).getWcs()",
                },
                "noise": {
                    "type": "RubinNoise",
                    "calexp": "$(@input.calexp)",
                },
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
                "weight": {
                    "file_name": wgt_pth,
                    "include_obj_var": include_obj_var,
                },
            },
        }

        img = galsim.Convolve(
            [
                galsim.Gaussian(fwhm=0.8),
                galsim.Exponential(half_light_radius=0.5, flux=flux),
            ]
        ).drawImage(
            nx=53,
            ny=53,
            wcs=wcs.local(galsim.PositionD(27, 27)),
        )

        for pth in [img_pth, wgt_pth]:
            try:
                os.remove(pth)
            except Exception:
                pass
        _cfg = copy.deepcopy(config)
        galsim.config.Process(_cfg)
        for pth in [img_pth, wgt_pth]:
            assert os.path.exists(pth)

        true_var_full = ((nse.sky_level[img.bounds] + img) / nse.gain[img.bounds]).array
        true_var_noimg_full = ((nse.sky_level[img.bounds]) / nse.gain[img.bounds]).array
        true_var = np.mean(true_var_full)
        true_var_noimg = np.mean(true_var_noimg_full)

        img_config = galsim.fits.read(img_pth).array
        var = np.var(img_config - img.array)

        print(
            "flux = %g, draw_method = %s, meas var = %g, var = %g, var no img = %g"
            % (flux, draw_method, var, true_var, true_var_noimg),
            flush=True,
        )

        assert_allclose(var, true_var, rtol=0.5 if flux > 1e5 else 0.1)

        if flux >= 1e10:
            assert not np.allclose(var, true_var_noimg, rtol=0.2)

        if flux == 0:
            # compare to returned sky var too
            # hack in some fields this expects and are normally there
            _cfg["file_num"] = 1
            _cfg["image_num"] = 2
            _cfg["image"]["noise"]["calexp"] = calexp
            ret_var = nse_build.getNoiseVariance(_cfg["image"]["noise"], _cfg)
            assert_allclose(var, ret_var, rtol=0.1)
            # the subimage is slightly different than the full image in its mean
            assert_allclose(ret_var, true_var_noimg, rtol=1e-4)

        wgt = galsim.fits.read(wgt_pth).array
        if draw_method == "fft" or (draw_method == "phot" and not include_obj_var):
            kwargs = {"rtol": 1e-6}
        else:
            # photons cause noise in the image part of the weight map
            kwargs = {"rtol": 0.05}
        if include_obj_var:
            assert_allclose(wgt, 1.0 / true_var_full, **kwargs)
        else:
            assert_allclose(wgt, 1.0 / true_var_noimg_full, **kwargs)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("sky_level", marks=pytest.mark.xfail),
        "sky_level_pixel",
    ],
)
def test_config_noise_raises(key):
    wcs_pth = os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
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
                "type": "Gaussian",
                "fwhm": 0.8,
            },
            "image": {
                "type": "Single",
                "wcs": {
                    "type": "RubinSkyWCS",
                    "wcs": "$(@input.calexp).getWcs()",
                },
                "noise": {
                    "type": "RubinNoise",
                    "calexp": "$(@input.calexp)",
                },
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }

        config["image"][key] = 1e3

        with pytest.raises(RuntimeError) as e:
            galsim.config.Process(copy.deepcopy(config))

        assert "sky level" in str(e.value)
        assert "RubinNoise" in str(e.value)


def test_config_noise_calexp_raises():
    calexp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
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
                "pixel_scale": 0.2,
                "noise": {
                    "type": "RubinNoise",
                },
                "size": 53,
            },
            "output": {
                "file_name": img_pth,
            },
        }

        with pytest.raises(RuntimeError) as e:
            _cfg = copy.deepcopy(config)
            galsim.config.Process(_cfg)
        assert "calexp" in str(e.value)
        assert "RubinNoise" in str(e.value)

        _cfg = copy.deepcopy(config)
        _cfg["calexp"] = calexp
        galsim.config.Process(_cfg)


def test_config_noise_current_var():
    calexp = lsst.afw.image.ExposureD.readFits(
        os.path.join(os.path.dirname(__file__), "data", "cexp.fits.fz")
    )
    nse = seacliff.RubinNoise(calexp)
    nse_build = RubinNoiseBuilder()
    base = {
        "modules": ["seacliff"],
        "calexp": calexp,
        "gal": {
            "type": "Exponential",
            "half_light_radius": 0.5,
            "flux": 0,
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
            "noise": {
                "type": "RubinNoise",
            },
            "size": 53,
        },
    }

    img = galsim.ImageD(53, 53)
    rng = galsim.BaseDeviate(24132)
    draw_method = "fft"
    logger = None
    current_var = 1e3
    img.addNoise(galsim.GaussianNoise(sigma=np.sqrt(1e3), rng=rng))

    base["file_num"] = 1
    base["image_num"] = 2

    rv = nse_build.addNoise(
        base["image"]["noise"],
        base,
        img,
        rng,
        current_var,
        draw_method,
        logger,
    )
    true_var_full = ((nse.sky_level[img.bounds] + img) / nse.gain[img.bounds]).array
    true_var = np.mean(true_var_full)
    var = np.var(img.array)

    assert_allclose(var, true_var, rtol=5e-2)
    assert_allclose(rv, true_var, rtol=5e-2)
    assert_allclose(var, rv, rtol=5e-2)

    # check that we error if too much is added
    with pytest.raises(RuntimeError) as e:
        img = galsim.ImageD(53, 53)
        rng = galsim.BaseDeviate(24132)
        draw_method = "fft"
        logger = None
        current_var = 5e3
        img.addNoise(galsim.GaussianNoise(sigma=np.sqrt(5e3), rng=rng))

        base["file_num"] = 1
        base["image_num"] = 2

        nse_build.addNoise(
            base["image"]["noise"],
            base,
            img,
            rng,
            current_var,
            draw_method,
            logger,
        )

    assert "Whitening/symmetrizing" in str(e.value)
    assert "RubinNoise" in str(e.value)
