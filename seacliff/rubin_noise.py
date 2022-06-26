import numpy as np


def get_rubin_skyvar_and_gain(calexp):
    """Return the map of the sky variance and the gain values from a calexp

    The algorithm here follows the PR
    https://github.com/lsst/meas_algorithms/pull/265/files which is based on code in
    Piff originally AFAIK. We also use the gains as reported by the code if possible
    following
    https://github.com/esheldon/metadetect/blob/master/metadetect/lsst/util.py#L239

    Parameters
    ----------
    calexp : lsst.afw.image.Exposure
        The background-subtacted calexp in ADU.

    Returns
    -------
    skyvar : np.ndarray
        The sky variance in units of ADU^2.
    gain : np.ndarray
        The gain in e-/AUD.
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
        #   <var in ADU> = m * <im in ADU> + b
        # the total variance in the image is Poisson noise for objects + sky
        # in electrons. This quantity is
        #   var for image in e- = (objects + sky) * gain
        # The objects in electrons are
        #   image in e- = objects * gain
        # Thus to get back to ADU, we scale the image by 1/gain and the variance by
        # 1/gain**2.
        # This gives us
        #  var in ADU = (objects + sky) / gain
        #  image in ADU = objects
        # Thus for the slope and intercept we have
        #  m = 1/gain
        #  b = sky / gain
        # Finally to get the sky variance we use the image itself and not the fitted sky
        # b:
        #   var - im / gain
        skyvar = calexp.variance.clone()
        gn = calexp.variance.clone()
        for amp_bbox in amp_bboxes:
            amp_im_arr = calexp.image[amp_bbox].array
            amp_var_arr = calexp.variance[amp_bbox].array
            good = np.isfinite(amp_var_arr) & np.isfinite(amp_im_arr)
            fit = np.polyfit(amp_im_arr[good], amp_var_arr[good], deg=1)
            gain = 1.0 / fit[0]
            skyvar[amp_bbox].array[good] -= amp_im_arr[good] / gain
            gn[amp_bbox].array[:, :] = gain
    else:
        skyvar = calexp.variance.clone()
        gn = calexp.variance.clone()
        for amp_bbox, gain in zip(amp_bboxes, gains):
            amp_im_arr = calexp.image[amp_bbox].array
            amp_var_arr = calexp.variance[amp_bbox].array
            good = np.isfinite(amp_var_arr) & np.isfinite(amp_im_arr)
            skyvar[amp_bbox].array[good] -= amp_im_arr[good] / gain
            gn[amp_bbox].array[:, :] = gain

    return skyvar.array.copy(), gn.array.copy()
