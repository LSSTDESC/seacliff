import galsim

from seacliff import RubinPSF


def _build_rubin_psf(config, base, ignore, gsparams, logger):
    opt = {"deconvolve_pixel": bool, "depixelize": bool, "wcs": None, "psf": None}
    params, safe = galsim.config.GetAllParams(
        config, base, opt=opt, ignore=ignore
    )

    if (
        "deconvolve_pixel" in params
        and "depixelize" in params
        and params["deconvolve_pixel"]
        and params["depixelize"]
    ):
        raise RuntimeError(
            "RubinPSF can only use deconvolve_pixel=True or depixelize=True, not both!"
        )

    if "psf" in params:
        rubin_psf = params.pop("psf")
    elif "calexp" in base:
        rubin_psf = base["calexp"].getPsf()
    else:
        raise RuntimeError(
            "RubinPSF could not find `psf` in base galsim config calexp or inputs!"
        )

    if "wcs" in params:
        wcs = params.pop("wcs")
    elif "wcs" in base:
        wcs = base["wcs"]
    else:
        raise RuntimeError(
            "RubinPSF could not find `wcs` in base galsim config or inputs!"
        )

    if 'image_pos' in base:
        image_pos = base['image_pos']
    else:
        raise RuntimeError("RubinPSF could not find `image_pos` in base galsim config!")

    if gsparams:
        gsparams = galsim.GSParams(**gsparams)
    else:
        gsparams = None

    galsim_psf = RubinPSF(rubin_psf, wcs)

    psf = galsim_psf.getPSF(
        image_pos,
        gsparams=gsparams,
        **params,
    )
    return psf, False


galsim.config.RegisterObjectType('RubinPSF', _build_rubin_psf)
