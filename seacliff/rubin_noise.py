import copy
import numpy as np

import galsim
from galsim.noise import BaseNoise
from galsim.random import PoissonDeviate

from seacliff.utils import mad


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


class RubinNoise(BaseNoise):
    """A galsim-compatible noise object that applies the noise from a Rubin calexp to
    a galsim image.

    The noise model in this class assumes Poisson fluctuations in electrons from both
    the sky and the objects. The number of electrons in each pixel is computed using
    the gain from the Rubin calexp. Thus any image to which this class is applied needs
    to be in ADU or "counts" units that match the Rubin ones. This can be achieved by
    using the proper zero-point to convert an image in physical units to Rubin counts.
    It should also not have a background. The output image has the noise applied.
    To set the variance on a Rubin calexp simulated with this class, use the sky_level
    and gain attributes like this:

        var = nse.sky_level + im / nse.gain

    where var is the total variance and im is the simulated image in ADU units.

    You can instantiate this object in one of several ways.

        1) Feed it a Rubin calexp:

            >>> nse = seacliff.RubinNoise(calexp)

        2) Feed it a galsim image with the variance (sky level) and gain maps:

            >>> nse = seacliff.RubinNoise(var, gain=gain)

        3) Feed it floats for the gain and/or sky level maps.

            >>> nse = seacliff.RubinNoise(10, gain=3)

    In all cases the units for the sky level are ADUs and the gain has units e-/ADU.

    Some of the code and methods here were lifted from galsim under its license:

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

    Parameters
    ----------
    calexp_or_sky_level : lsst.afw.image.Exposure, galsim.Image, or float
        A Rubin calexp from which to extract the sky level and gain, an image
        of the sky level in a Rubin calexp, or a float value with the sky level.
    gain : galsim.Image, float, or None, optional
        The gain as an image or a float value. Only allowed if `calexp_or_sky_level` is
        not a Rubin calexp.
    rng : galsim.BaseDeviate or None, optional
        An RNG instance to use for generating noise.
    mad_clipping : float or None,
        If given, the sky level will be clipped to be at most this many median
        absolute deviations from the median sky level. Useful for truncating large
        values in the sky level extracted from a calexp.

    Attributes
    ----------
    rng : galsim.BaseDeviate
        The current `galsim.BaseDeviate` attached to this class.
    sky_level : galsim.Image or float.
        An image of the sky level in ADU for the calexp.
    gain : galsim.Image or float.
        An image of the gain in e-/ADU for the calexp.
    mad_clipping : float or None
        If not None, the level of MAD clipping applied to the sky level.
    """

    def __init__(self, calexp_or_sky_level, gain=None, rng=None, mad_clipping=None):
        super().__init__(rng)
        self._pd = PoissonDeviate(self.rng)
        self._mad_clipping = mad_clipping

        if gain is None:
            # we got a calexp
            sv, gn = get_rubin_skyvar_and_gain(calexp_or_sky_level)
            self._gn = galsim.ImageD(gn)
        else:
            # we got a sky variance and gain
            self._gn = gain
            if isinstance(calexp_or_sky_level, galsim.Image):
                sv = calexp_or_sky_level.array.copy()
            else:
                sv = calexp_or_sky_level

        if np.ndim(sv) > 0:
            if self.mad_clipping is not None:
                sd, mn = mad(sv, return_median=True)
                msk = np.abs(sv - mn) > mad_clipping * sd
                if np.any(msk):
                    sgn = np.sign(sv - mn)
                    sv[msk] = mn + sgn[msk] * mad_clipping * sd
            self._sv = galsim.ImageD(sv)
        else:
            self._sv = sv

    @property
    def mad_clipping(self):
        """MAD clipping # of sigmas used on the sky level."""
        return self._mad_clipping

    @property
    def sky_level(self):
        """calexp sky level in units of ADU or 'counts'"""
        return self._sv

    @property
    def gain(self):
        """calexp gain in units of e-/ADU"""
        return self._gn

    def _applyTo(self, image):
        if (
            isinstance(self.sky_level, galsim.Image)
            and image.bounds != self.sky_level.bounds
        ) or (isinstance(self.gain, galsim.Image) and image.bounds != self.gain.bounds):
            raise RuntimeError(
                "The image must have the same pixel bounds as the sky level and gain!"
            )

        # the noise array is in ADU
        noise_array = np.empty(np.prod(image.array.shape), dtype=float)
        noise_array[:] = image.array.flatten()

        # Minor subtlety for integer images. It's a bit more consistent to convert to an
        # integer with the sky still added and then subtract off the sky. But this isn't
        # quite right if the sky has a fractional part.  So only subtract off the
        # integer part of the sky at the end. For float images, you get the same answer
        # either way, so it doesn't matter.
        if isinstance(self.sky_level, galsim.Image):
            sv = self.sky_level.array.flatten()
        else:
            sv = self.sky_level

        frac_sky = sv - image.dtype(sv)
        int_sky = sv - frac_sky

        if np.any(sv != 0):
            noise_array += sv

        # The noise_image now has the expectation values for each pixel with the sky
        # added but in ADU.
        # We convert to e-, make the random draw, then convert back.
        if isinstance(self.gain, galsim.Image):
            gn = self.gain.array.flatten()
        else:
            gn = self.gain

        noise_array *= gn
        noise_array = noise_array.clip(0)  # Make sure no negative values
        self._pd.generate_from_expectation(noise_array)
        noise_array /= gn

        # Subtract off the sky, since we don't want it in the final image.
        if np.any(frac_sky != 0):
            noise_array -= frac_sky
        # Noise array is now the correct value for each pixel.
        np.copyto(image.array, noise_array.reshape(image.array.shape), casting="unsafe")
        if np.any(int_sky != 0):
            if np.ndim(int_sky) > 0:
                int_sky = int_sky.reshape(image.array.shape)
            image -= int_sky

    def _getVariance(self):
        return self.sky_level / self.gain

    def _withVariance(self, variance):
        isarr = False

        if isinstance(variance, galsim.Image):
            _variance = variance.array
            isarr = True
        else:
            _variance = variance

        if isinstance(self.gain, galsim.Image):
            _gain = self.gain.array
            isarr = True
        else:
            _gain = self.gain

        vg = _variance * _gain

        return RubinNoise(
            galsim.ImageD(vg) if isarr else vg,
            gain=self.gain,
            rng=self.rng,
            mad_clipping=None,
        )

    def _withScaledVariance(self, variance_ratio):
        return RubinNoise(
            self.sky_level * variance_ratio,
            gain=self.gain,
            rng=self.rng,
            mad_clipping=None,
        )

    def copy(self, rng=None, memo=None):
        """Make a copy of this noise object.

        By default, the copy will share the `BaseDeviate` random number generator
        with the RubinNoise from which the copy was made. However, you can provide a
        new rng for the copy if you'd like.

        Parameters
        ----------
        rng : galsim.BaseDeviate or None, optional
            An RNG instance to use for generating noise for the copy. If not given,
            the copy will share its RNG with the original object.

        Returns
        -------
        Copy of this object.
        """
        if rng is None:
            rng = self.rng

        if memo is not None:
            return RubinNoise(
                copy.deepcopy(self.sky_level, memo),
                gain=copy.deepcopy(self.gain, memo),
                rng=rng,
                mad_clipping=None,
            )
        else:
            return RubinNoise(
                copy.deepcopy(self.sky_level),
                gain=copy.deepcopy(self.gain),
                rng=rng,
                mad_clipping=None,
            )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(memo=memo)

    def __repr__(self):
        return "seacliff.RubinNoise(%r, gain=%r, rng=%r, mad_clipping=%r)" % (
            self.sky_level,
            self.gain,
            self.rng,
            None,
        )

    def __str__(self):
        return "seacliff.RubinNoise(%r, gain=%r, mad_clipping=%r)" % (
            self.sky_level,
            self.gain,
            None,
        )
