import numpy as np

from seacliff.utils import mad


def test_mad():
    rng = np.random.RandomState(seed=10)
    x = rng.normal(scale=1.5, size=10_000_000)

    np.testing.assert_allclose(mad(x), 1.5, atol=1e-3, rtol=1e-3)

    rng = np.random.RandomState(seed=10)
    x = rng.normal(scale=1.5, size=(10_000_000, 2))
    md = mad(x, axis=0)
    assert md.shape == (2,)
    np.testing.assert_allclose(md, 1.5, atol=1e-3, rtol=1e-3)

    rng = np.random.RandomState(seed=10)
    x = rng.normal(scale=1.5, size=10_000_000)
    _, md = mad(x, return_median=True)
    np.testing.assert_allclose(md, np.median(x))
