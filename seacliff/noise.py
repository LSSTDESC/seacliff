import numpy as np


def get_rubin_skyvar_and_gain(calexp):
    """Return the map of the sky variance and the gain values from a calexp

    The algorithm here follows the PR
    https://github.com/lsst/meas_algorithms/pull/265/files which is based on code in
    Piff originally AFAIK.

    Parameters
    ----------
    calexp : lsst.afw.image.Exposure
        The background-subtacted calexp.

    Returns
    -------
    skyvar : np.ndarray
        The sky variance in units of <something in pixel counts>/gain**2.
    gain : np.ndarray
        The map of gains that converts counts to calexp units
        `<calexp units> = <something in pixel counts>/gain`.
    """
    try:
        amps = calexp.getDetector().getAmplifiers()
        amp_bboxes = [amp.getBBox() for amp in amps]
    except AttributeError:
        amp_bboxes = [calexp.getBBox()]

    try:
        amps = calexp.getDetector().getAmplifiers()
        gains = [amp.getGain() for amp in amps]
    except AttributeError:
        gains = None

    if gains is None or len(gains) != len(amp_bboxes):
        # the gains do not match or are not there so we have to fit for it
        # the fit here is
        #   var = m * im + b
        # the total variance in the image is Poisson noise for objects + sky
        # this variance is scaled by the gain to match the image scaling
        # to get (objects + sky) / gain**2
        # the image is background subtracted and scaled so im = objects / gain
        # thus the slope is m = 1/gain and the intercept has to be b = sky / gain**2
        # we use the estimate of the sky from the image-variance subtracted variance
        # plane
        # (e.g., raw var - im / gain) as opposed to the intercept of the fit.
        skyvar = calexp.variance_plane.clone()
        gn = calexp.variance_plane.clone()
        for amp_bbox in amp_bboxes:
            amp_im_arr = calexp[amp_bbox].image.array
            amp_var_arr = calexp.variance_plane[amp_bbox].array
            good = (
                (amp_var_arr != 0)
                & np.isfinite(amp_var_arr)
                & np.isfinite(amp_im_arr)
            )
            fit = np.polyfit(amp_im_arr[good], amp_var_arr[good], deg=1)
            gain = 1.0 / fit[0]
            skyvar[amp_bbox].array[good] -= amp_im_arr[good] / gain
            gn[amp_bbox].array[:, :] = gain
    else:
        skyvar = calexp.variance_plane.clone()
        gn = calexp.variance_plane.clone()
        for amp_bbox, gain in zip(amp_bboxes, gains):
            amp_im_arr = calexp[amp_bbox].image.array
            amp_var_arr = calexp.variance_plane[amp_bbox].array
            good = (
                (amp_var_arr != 0)
                & np.isfinite(amp_var_arr)
                & np.isfinite(amp_im_arr)
            )
            skyvar[amp_bbox].array[good] -= amp_im_arr[good] / gain
            gn[amp_bbox].array[:, :] = gain

    return skyvar.array.copy(), gn.array.copy()
