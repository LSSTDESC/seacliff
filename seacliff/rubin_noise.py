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

    You can instantiate this object in one of two ways.

        1) Feed it a Rubin calexp:

            >>> nse = seacliff.RubinNoise(calexp)

        2) Feed it a galsim image with the variance (sky level) and gain maps:

            >>> nse = seacliff.RubinNoise(var, gain=gain)

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
    calexp_or_sky_level :
    gain :
    rng :
    mad_clipping :

    Attributes
    ----------
    rng :
    sky_level :
    gain :
    mad_clipping :
    """

    def __init__(self, calexp_or_sky_level, gain=None, rng=None, mad_clipping=None):
        super().__init__(rng)
        self._pd = PoissonDeviate(self.rng)
        self._mad_clipping = mad_clipping

        if gain is not None:
            # we got a sky variance and gain
            self._gn = gain
            sv = calexp_or_sky_level.array.copy()
        else:
            # we got a calexp
            sv, gn = get_rubin_skyvar_and_gain(calexp_or_sky_level)
            self._gn = galsim.ImageD(gn)

        if self.mad_clipping is not None:
            sd, mn = mad(sv, return_median=True)
            msk = np.abs(sv - mn) > mad_clipping * sd
            if np.any(msk):
                sgn = np.sign(sv - mn)
                sv[msk] = mn + sgn[msk] * mad_clipping * sd

        if gain is not None:
            _sv = calexp_or_sky_level.copy()
            _sv.array[:, :] = sv
            self._sv = _sv
        else:
            self._sv = galsim.ImageD(sv)

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
        if image.bounds != self.sky_level.bounds or image.bounds != self.gain.bounds:
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
        frac_sky = self.sky_level.array - image.dtype(self.sky_level.array)
        int_sky = self.sky_level.array - frac_sky

        if np.any(self.sky_level.array != 0):
            noise_array += self.sky_level.array.flatten()

        # The noise_image now has the expectation values for each pixel with the sky
        # added but in ADU.
        # We convert to e-, make the random draw, then convert back.
        noise_array *= self.gain.array.flatten()
        noise_array = noise_array.clip(0)  # Make sure no negative values
        self._pd.generate_from_expectation(noise_array)
        noise_array /= self.gain.array.flatten()

        # Subtract off the sky, since we don't want it in the final image.
        if np.any(frac_sky != 0):
            noise_array -= frac_sky.flatten()
        # Noise array is now the correct value for each pixel.
        np.copyto(image.array, noise_array.reshape(image.array.shape), casting="unsafe")
        if np.any(int_sky != 0):
            image -= int_sky

    def _getVariance(self):
        return self.sky_level / self.gain

    def _withVariance(self, variance):
        if not isinstance(variance, galsim.Image):
            _variance = self.sky_level.copy()
            _variance.array[:, :] = variance
        else:
            _variance = variance

        return RubinNoise(
            _variance * self.gain,
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
        rng :

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
