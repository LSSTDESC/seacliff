# these imports define the namespace and runs some galsim registration stuff for the
# config parser
from . import input  # noqa

# these variables are always available for eval
import galsim.config.value_eval

# this is always available
galsim.config.value_eval.eval_base_variables += ["calexp"]
