# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy

import numpy as np


def check_pickle_eval_repr_copy(obj1, func=None, irreprable=False, extra_tests=None):
    """test objects in a standard way - modified from galsim for seacliff"""
    if func is None:

        def func(x):
            return x

    # In case the repr uses these:
    from numpy import (  # noqa
        array,
        uint16,
        uint32,
        int16,
        int32,
        float32,
        float64,
        complex64,
        complex128,
        ndarray,
    )
    from astropy.units import Unit  # noqa
    import galsim  # noqa
    import seacliff  # noqa
    from numbers import Integral, Real, Complex  # noqa
    import lsst.geom  # noqa
    import lsst.afw.geom  # noqa

    # check that it pickles
    obj2 = pickle.loads(pickle.dumps(obj1))
    assert obj2 is not obj1
    if extra_tests is not None:
        extra_tests(obj1, obj2)

    # check possible transformations of it
    f1 = func(obj1)
    f2 = func(obj2)
    assert f1 == f2
    if extra_tests is not None:
        extra_tests(f1, f2)

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    try:
        from collections.abc import Hashable
    except ImportError:
        from collections import Hashable
    if isinstance(obj1, Hashable):
        assert hash(obj1) == hash(obj2)

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    if extra_tests is not None:
        extra_tests(obj1, obj3)
    random = (
        hasattr(obj1, "rng")
        or isinstance(obj1, galsim.BaseDeviate)
        or "rng" in repr(obj1)
    )
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1
        if extra_tests is not None:
            extra_tests(f1, f3)
    elif isinstance(obj1, galsim.BaseDeviate):
        f1 = func(obj1)  # But BaseDeviates will be ok.  Just need to remake f1.
        f3 = func(obj3)
        assert f3 == f1
        if extra_tests is not None:
            extra_tests(f1, f3)

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    if extra_tests is not None:
        extra_tests(obj1, obj4)
    f4 = func(obj4)
    if random:
        f1 = func(obj1)
    assert f4 == f1  # But everything should be identical with deepcopy.
    if extra_tests is not None:
        extra_tests(f1, f4)

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these,
    # we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since
    # these don't
    # respect the eval/repr roundtrip.

    if not random and not irreprable:
        # A further complication is that the default numpy print options do not
        # lead to sufficient
        # precision for the eval string to exactly reproduce the original object,
        # and start
        # truncating the output for relatively small size arrays.  So we temporarily
        # bump up the
        # precision and truncation threshold for testing.
        with galsim.utilities.printoptions(precision=20, threshold=np.inf):
            obj5 = eval(repr(obj1))
        if extra_tests is not None:
            extra_tests(obj1, obj5)

        f5 = func(obj5)
        assert f5 == f1, "func(obj1) = %r\nfunc(obj5) = %r" % (f1, f5)
        if extra_tests is not None:
            extra_tests(f1, f5)
    else:
        # Even if we're not actually doing the test, still make the repr to check
        # for syntax errors.
        repr(obj1)
