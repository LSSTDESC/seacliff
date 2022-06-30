import galsim.config
from seacliff import RubinSkyWCS


class RubinSkyWCSBuilder(galsim.config.SimpleWCSBuilder):
    """Build a RubinSkyWCS"""

    def getKwargs(self, build_func, config, base):
        req = {"wcs": None}
        opt = {"origin": galsim.Position}
        kwargs, _ = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        return kwargs


galsim.config.RegisterWCSType("RubinSkyWCS", RubinSkyWCSBuilder(RubinSkyWCS))
