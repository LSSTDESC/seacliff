import galsim.config
from seacliff import RubinSkyWCS


class RubinSkyWCSBuilder(galsim.config.SimpleWCSBuilder):
    """Build a RubinSkyWCS"""

    def getKwargs(self, build_func, config, base):
        req = {"wcs": None}
        kwargs, _ = galsim.config.GetAllParams(config, base, req=req)
        return kwargs


galsim.config.RegisterWCSType("RubinSkyWCS", RubinSkyWCSBuilder(RubinSkyWCS))
