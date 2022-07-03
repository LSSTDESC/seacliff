import galsim.config
from seacliff import RubinSkyWCS


class RubinSkyWCSBuilder(galsim.config.SimpleWCSBuilder):
    """Build a RubinSkyWCS"""

    def getKwargs(self, build_func, config, base):
        opt = {"wcs": None}
        kwargs, _ = galsim.config.GetAllParams(config, base, opt=opt)

        if "wcs" not in kwargs and "calexp" in base:
            kwargs["wcs"] = base["calexp"].getWcs()

        if "wcs" not in kwargs:
            raise RuntimeError(
                "RubinWCS could not find `wcs` in input or base galsim config!"
            )

        return kwargs


galsim.config.RegisterWCSType("RubinSkyWCS", RubinSkyWCSBuilder(RubinSkyWCS))
