import galsim.config
import numpy as np
from galsim.config import NoiseBuilder, PoissonNoise, noise_ignore

from seacliff import RubinNoise


class RubinNoiseBuilder(NoiseBuilder):
    def _get_params(self, config, base):
        sky_pars = ["sky_level", "sky_level_pixel"]
        if any(p in base["image"] for p in sky_pars):
            raise RuntimeError(
                "The sky level cannot be given in image when using RubinNoise!"
            )

        opt = {"mad_clipping": float, "calexp": None}
        params = galsim.config.GetAllParams(
            config,
            base,
            opt=opt,
            ignore=noise_ignore,
        )[0]

        if "calexp" not in params and "calexp" in base:
            params["calexp"] = base["calexp"]

        if "calexp" not in params:
            raise RuntimeError(
                "RubinNoise could not find `calexp` in either "
                "image.noise or base galsim config!"
            )

        return params

    def _get_rubin_noise(self, config, base):
        pars = self._get_params(config, base)

        # we cache this object using the following key since it may involve a fit
        # to the calexp image and variance plane
        tag = (
            id(pars["calexp"]),
            pars.get("mad_clipping", None),
            id(base),
            base["file_num"],
            base["image_num"],
        )
        if (
            config.get("_current_noise_tag", None) != tag
            or "_current_noise" not in config
            or "_current_noise_var" not in config
        ):
            rnse = RubinNoise(
                pars["calexp"],
                mad_clipping=pars.get("mad_clipping", None),
            )
            # the noise is always in ADU units which is (sky * gain) / gain**2
            ret_var_full = rnse.sky_level / rnse.gain
            ret_var = np.mean(ret_var_full.array)

            config["_current_noise_tag"] = tag
            config["_current_noise"] = rnse
            config["_current_noise_var"] = ret_var
            config["_current_noise_var_full"] = ret_var_full

        return (
            config["_current_noise"],
            config["_current_noise_var"],
            config["_current_noise_var_full"],
        )

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):
        rnse, ret_var, var_full = self._get_rubin_noise(config, base)

        # existing variance is removed from the sky level
        if current_var:
            if np.any(var_full.array < current_var):
                raise RuntimeError(
                    "Whitening/symmetrizing already added more noise than the sky "
                    "for the RubinNoise model!"
                )

            # make a new noise object with the sky level reduced
            rnse = rnse.withVariance(var_full - current_var)

        # if we are photon shooting, then the variance from objects is in the image
        # already so we do not add that.
        if draw_method == "phot":
            bnds = im.bounds
            # make an image in electrons
            noise_im = rnse.sky_level[bnds].copy() * rnse.gain[bnds]
            # poisson sample
            noise_im.addNoise(PoissonNoise(rng))
            # convert back to adu
            noise_im /= rnse.gain[bnds]
            # remove mean sky
            noise_im -= rnse.sky_level[bnds]
            im += noise_im
        else:
            im.addNoise(rnse)

        return ret_var

    def getNoiseVariance(self, config, base, full=False):
        _, ret_var, var_full = self._get_rubin_noise(config, base)
        if full:
            bounds = base["current_noise_image"].bounds
            return var_full[bounds].copy()
        else:
            return ret_var

    def addNoiseVariance(self, config, base, im, include_obj_var, logger):
        rnse, _, _ = self._get_rubin_noise(config, base)
        if include_obj_var:
            bounds = base["current_noise_image"].bounds
            im += base["current_noise_image"] / rnse.gain[bounds]

        im += self.getNoiseVariance(config, base, full=True)


galsim.config.RegisterNoiseType("RubinNoise", RubinNoiseBuilder())
