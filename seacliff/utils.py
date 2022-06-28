import numpy as np


def mad(x, axis=None, return_median=False):
    """
    median absolute deviation - scaled like a standard deviation

        mad = 1.4826 * median(|x-median(x)|)

    Parameters
    ----------
    x : array-like
        Array to take MAD of.
    axis : int
        Axis over which to take MAD. Default of None indicates all axes.
    return_median : bool
        Return the median as well.

    Returns
    -------
    mad : float
        MAD of array x
    med : float or array
        If `return_median` is True, return the median. May be an array in `axis`
        keyword indicates a reduction not along all of the axes of the input.
    """
    kd = True if axis is not None else False
    med = np.median(x, axis=axis, keepdims=kd)
    mad = np.median(np.abs(x - med), axis=axis)
    if return_median:
        return 1.4826 * mad, med
    else:
        return 1.4826 * mad
