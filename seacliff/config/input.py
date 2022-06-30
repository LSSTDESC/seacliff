import galsim.config
import lsst.afw.image


def _read_calexp(dtype="double", *, file_name, logger):
    logger.error("Reading calexp '%s' with dtype '%s'", file_name, dtype)
    if dtype == "double":
        return lsst.afw.image.ExposureD.readFits(file_name)
    elif dtype == "float":
        return lsst.afw.image.ExposureF.readFits(file_name)
    elif dtype == "int":
        return lsst.afw.image.ExposureI.readFits(file_name)
    elif dtype == "long":
        return lsst.afw.image.ExposureL.readFits(file_name)
    else:
        raise RuntimeError("lsst.afw.image dtype '%s' not recognized!", dtype)


class CalexpLoader(galsim.config.InputLoader):
    """Load a calexp from a file.

    It uses yaml config like this:

        ```yaml
        input:
          calexp:
            file_name: /path/to/calexp.fits  # required
            # this optional field sets the dtype of the input calexp
            # dtype: double  # one of double, float, int or long, default is double
        ```
    """
    def getKwargs(self, config, base, logger):
        req = {"file_name": str}
        opt = {"dtype": str}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        if self.takes_logger:
            kwargs['logger'] = logger
        return kwargs, True

    def initialize(self, input_objs, num, base, logger):
        if len(input_objs) == 1:
            base["calexp"] = input_objs[0]
        else:
            base["calexp"] = input_objs


galsim.config.RegisterInputType(
    'calexp',
    CalexpLoader(_read_calexp, file_scope=True, takes_logger=True),
)
