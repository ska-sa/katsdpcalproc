#
# Updated refrence pointing script
# This script contains low level functions to calculate
# offsets of a reference pointing observation
# The following low-level functions are used to calculate the offsets:
# Extract gains per pointing offset, per receptor and per frequency chunk.
# Fit primary beams to the gains
# Calculate offsets from fitted beams
# Ludwig Schwardt & Tasmiyah Rawoot
#

import logging

import numpy as np

from astropy.modeling import models, fitting
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from katpoint import rad2deg, deg2rad, lightspeed, wrap_angle, RefractionCorrection


logger = logging.getLogger(__name__)


def get_offset_gains(bp_gains, offsets, ants, channel_freqs, pols, num_chunks=16):
    """Extract gains per pointing offset, per receptor and per frequency chunk.

    Parameters
    ----------
    bp_gains : numpy.array
        Numpy array of shape (n_offsets, n_channels, n_polarizations, n_antennas)
        containing bandpass gains
    offsets : list of sequences of 2 floats
        list of requested (x, y) pointing offsets co-ordinates relative to target,
        in degrees.
    ants : list of :class:`katpoint.Antenna`
        A list of antenna objects for fitted beams
    channel_freqs : list of floats
        A list of channel frequencies
    pols : list of strings
        A list containing polarisations, eg, ["h","v"]
    num_chunks : int, optional
        Group the frequency channels into this many sections to
        obtain pointing fits

    Returns
    -------
    data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
        Complex gains per receptor, as multiple records per offset and
        frequency chunk ie. len(data_points)=63, len(data_points[i])=
        num_chunks*n_offsets, len(data_points[i][j]=5)
    """
    if bp_gains.shape != (len(offsets), len(channel_freqs), len(pols), len(ants)):
        raise ValueError(
            "bp_gains must have shape (n_offsets, n_channels, n_polarizations, n_antennas)")
    # Calculating chunk frequencies
    chunk_freqs = channel_freqs.reshape(num_chunks, -1).mean(axis=1)
    data_points = {}
    for offset, offset_bp_gain in zip(offsets, bp_gains):
        for a, ant in enumerate(ants):
            pol_gain = np.zeros(num_chunks)
            pol_weight = np.zeros(num_chunks)
            # Iterate over polarisations (effectively over inputs)
            for pol in range(len(pols)):
                inp = ant.name + pols[pol]
                bp_gain = offset_bp_gain[:, pol, a]
                if bp_gain is None:
                    continue
                masked_gain = np.ma.masked_invalid(bp_gain)
                abs_gain_chunked = np.abs(masked_gain).reshape(num_chunks, -1)
                abs_gain_mean = abs_gain_chunked.mean(axis=1)
                abs_gain_std = abs_gain_chunked.std(axis=1)
                abs_gain_var = abs_gain_std.filled(np.inf) ** 2
                # Replace any zero variance with the smallest non-zero variance
                # across chunks, but if all are zero it is fishy and ignored.
                zero_var = abs_gain_var == 0.
                if all(zero_var):
                    abs_gain_var = np.ones_like(abs_gain_var) * np.inf
                else:
                    abs_gain_var[zero_var] = abs_gain_var[~zero_var].min()
                # Number of valid samples going into statistics
                abs_gain_N = (~abs_gain_chunked.mask).sum(axis=1)
                # Generate standard precision weights based on empirical stdev
                abs_gain_weight = abs_gain_N / abs_gain_var
                # Prepare some debugging output
                stats_mean = ' '.join("%4.2f" % (m,) for m in
                                      abs_gain_mean.filled(np.nan))
                stats_std = ' '.join("%4.2f" % (s,) for s in
                                     abs_gain_std.filled(np.nan))
                stats_N = ' '.join("%4d" % (n,) for n in abs_gain_N)
                bp_mean = np.nanmean(np.abs(bp_gain))
                logger.debug("%s %s %4.2f mean | %s",
                             tuple(offset), inp, bp_mean, stats_mean)
                logger.debug("%s %s %4.2f std  | %s",
                             tuple(offset), inp, bp_mean, stats_std)
                logger.debug("%s %s      N    | %s",
                             tuple(offset), inp, stats_N)
                # Blend new gains into existing via weighted averaging.
                # XXX We currently combine HH and VV gains at the start to get
                # Stokes I gain but in future it might be better to fit
                # separate beams to HH and VV.
                pol_gain, pol_weight = np.ma.average(
                    np.c_[pol_gain, abs_gain_mean], axis=1,
                    weights=np.c_[pol_weight, abs_gain_weight], returned=True)
            if pol_weight.sum() > 0:
                # Turn masked values into NaNs pre-emptively to avoid warning
                # when recarray in beam fitting routine forces this later on.
                pol_gain = pol_gain.filled(np.nan)
                data = data_points.get(a, [])
                for freq, gain, weight in zip(chunk_freqs, pol_gain, pol_weight):
                    data.append((offset[0], offset[1], freq, gain, weight))
                data_points[a] = data
    return data_points


class BeamPatternFit:
    """Fit analytic beam pattern to power or gain data defined on 2-D plane.

    This fits a two-dimensional Gaussian curve to power or voltage gain data
    as a function of 2-D coordinates. The Gaussian function is restricted to
    have a diagonal covariance matrix; in other words, its elliptical contours
    are aligned with the axes. The Gaussian bump represents an antenna beam
    pattern convolved with a point source.

    Parameters
    ----------
    center : sequence of 2 floats
        Initial guess of 2-element beam center, in target coordinate units
    width : sequence of 2 floats
        Initial guess of 2-element beamwidth vector, expressed as FWHM in
        units of target coordinates
    height : float
        Initial guess of beam pattern amplitude or height

    Attributes
    ----------
    expected_width : array of float, shape (2,)
        Initial guess of beamwidth, saved as expected width for checks
    is_valid : bool
        True if beam parameters are within reasonable ranges after fit
    std_center : array of float, shape (2,) or None
        Standard error of beam center, only set after :func:`fit`
    std_width : array of float, shape (2,) or None
        Standard error of beamwidth(s), only set after :func:`fit`
    std_height : float or None
        Standard error of beam height, only set after :func:`fit`
    """

    def __init__(self, center, width, height):
        width = np.asarray(width)
        std = width / GAUSSIAN_SIGMA_TO_FWHM
        model = models.Gaussian2D(height, center[0], center[1], std[0], std[1])
        # Fix theta = 0 to ensure that elliptical beam contours line up with axes
        model.theta.fixed = True
        self._set_model(model)
        self._fit = fitting.LevMarLSQFitter(calc_uncertainties=True)
        self.expected_width = width
        self.is_valid = False
        self.std_center = self.std_width = self.std_height = None

    def _set_model(self, model):
        """Set underlying model and unpack center, width and height."""
        self._model = model
        self.center = np.array([model.x_mean.value, model.y_mean.value])
        self.width = np.array([model.x_fwhm, model.y_fwhm])
        self.height = model.amplitude.value

    def fit(self, x, y, std_y=1.0):
        """Fit a beam pattern to power or voltage gain data.

        The center, width and height of the fitted beam pattern (and their
        standard errors) can be obtained from the corresponding member
        variables after this is run.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Sequence of 2-dimensional target coordinates (as column vectors)
        y : array-like, shape (N,)
            Sequence of corresponding power or voltage gain values to fit
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`
        """
        new_model = self._fit(self._model, x=x[0], y=x[1], z=y, weights=1.0 / std_y)
        self._set_model(new_model)
        param_cov = self._fit.fit_info['param_cov']
        # The parameter cov matrix is absent if singular: be very uncertain then
        # XXX Only an issue with LevMarLSQFitter - remove check for LMLSQFitter
        if param_cov is None:
            param_std = np.full(5, np.inf)
        else:
            param_std = np.sqrt(np.diag(param_cov))
        self.std_center = param_std[1:3]
        self.std_width = GAUSSIAN_SIGMA_TO_FWHM * param_std[3:5]
        self.std_height = param_std[0]
        self.is_valid = all(np.isfinite(self.center)) and self.height > 0.0
        # Also invalidate beam if fit succeeds but there are no uncertainties
        self.is_valid &= all(np.isfinite(self.std_center))
        # XXX: POTENTIAL TWEAK
        norm_width = self.width / self.expected_width
        self.is_valid &= all(norm_width > 0.9) and all(norm_width < 1.25)

    def __call__(self, x):
        """Evaluate fitted beam pattern function on new target coordinates.

        Parameters
        ----------
        x : array-like, shape (2, M)
            Sequence of 2-dimensional target coordinates (as column vectors)

        Returns
        -------
        y : array, shape (M,)
            Sequence of total power values representing fitted beam
        """
        return self._model(x[0], x[1])


def _voltage_beamwidths(beamwidth_factor, frequency, dish_diameter):
    """Helper function to calculate expected width
    Parameters
    ----------

    beamwidth_factor : float
        Power beamwidth of antenna
    frequency : float
        Frequency at which to calculate volatge beamwidth
    dish_diameter : float
        Diameter of antenna dish
    Returns
    -------
    expected_width : sequence of 2 floats, or float
        Initial guess of single beamwidth for both dimensions, or 2-element
        beamwidth vector, expressed as FWHM in units of target coordinates

    """
    expected_width = rad2deg(beamwidth_factor * lightspeed
                             / frequency / dish_diameter)
    # Convert power beamwidth to gain / voltage beamwidth
    expected_width = np.sqrt(2.0) * expected_width
    # XXX This assumes we are still using default ant.beamwidth of 1.22
    # and also handles larger effective dish diameter in H direction
    expected_width = (0.8 * expected_width, 0.9 * expected_width)
    return expected_width


def beam_fit(data_points, ants, num_chunks=16, beam_center=(0.5, 0.5)):
    """Fit primary beams to receptor gains obtained at various offset pointings.

    Parameters
    ----------

    data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
        Complex gains per receptor, as multiple records per offset and
        frequency chunk
    ants : list of :class:`katpoint.Antenna`
        A list of antenna objects for fitted beams
    num_chunks : int, optional
        Group the frequency channels into this many sections to obtain
        pointing fits
   beam_center : sequence of 2 floats
        Initial guess of 2-element beam center, in target coordinate units
    Returns
    -------
    beams : dict mapping receptor name to list of :class:`BeamPatternFit`
        Fitted primary beams, per receptor and per frequency chunk

    """
    beams = {}
    # Iterate over receptors
    for a in data_points:
        data = np.rec.fromrecords(data_points[a], names='x,y,freq,gain,weight')
        data = data.reshape(-1, num_chunks)
        ant = ants[a]
        # Iterate over frequency chunks but discard typically dodgy band edges
        for chunk in range(1, num_chunks - 1):
            chunk_data = data[:, chunk]
            is_valid = np.nonzero(~np.isnan(chunk_data['gain']) & (chunk_data['weight'] > 0.))[0]
            chunk_data = chunk_data[is_valid]
            if len(chunk_data) == 0:
                continue
            # expected widths for each frequency channel
            expected_width = _voltage_beamwidths(ant.beamwidth, chunk_data['freq'][0], ant.diameter)
            beam = BeamPatternFit(beam_center, expected_width, 1.0)
            x = np.c_[chunk_data['x'], chunk_data['y']].T
            y = chunk_data['gain']
            std_y = np.sqrt(1. / chunk_data['weight'])
            try:
                beam.fit(x, y, std_y)
            except TypeError:
                continue
            beamwidth_norm = beam.width / np.array(expected_width)
            center_norm = beam.center / beam.std_center
            logger.debug("%s %2d %2d: height=%4.2f width=(%4.2f, %4.2f) "
                         "center=(%7.2f, %7.2f)%s",
                         ant.name, chunk, len(y), beam.height,
                         beamwidth_norm[0], beamwidth_norm[1],
                         center_norm[0], center_norm[1],
                         ' X' if not beam.is_valid else '')
            # Store beam per frequency chunk and per receptor
            beams_freq = beams.get(ant.name, [None] * num_chunks)
            beams_freq[chunk] = beam
            beams[ant.name] = beams_freq
    return beams


def calc_pointing_offsets(ants, middle_time, temperature, humidity, pressure,
                          beams, target, existing_az_el_adjust=None):
    """Calculate pointing offsets per receptor based on primary beam fits.

    Parameters
    ----------
    ants: list of :class:`katpoint.Antenna`
        A list of antenna objects for fitted beams
    middle_time : float
        Unix timestamp at the middle of sequence of offset pointings, used to
        find the mean location of a moving target (and reference for weather)
    temperature, humidity, pressure : :class: 'float'
        Atmospheric conditions at middle time, used for refraction correction
    beams : dict mapping receptor name to list of :class:`BeamPatternFit`
        Fitted primary beams, per receptor and per frequency chunk
    target : :class:`katpoint.Target` object
        The target on which offset pointings were done
    existing_az_el_adjust : array of float, shape(n_ants, 2)
        Numpy array of existing (az,el) adjustment of target for
        each antenna. Defaults to zero

    Returns
    -------
    pointing_offsets : dict mapping receptor name to offset data (10 floats)
        Pointing offsets per receptor in degrees, stored as a sequence of
          - requested (az, el) after refraction (input to the pointing model),
          - full (az, el) offset, including contributions of existing pointing
            model, any existing adjustment and newly fitted adjustment
            (useful for fitting new pointing models as it is independent),
          - full (az, el) adjustment on top of existing pointing model,
            replacing any existing adjustment (useful for reference pointing),
          - relative (az, el) adjustment on top of existing pointing model and
            adjustment (useful for verifying reference pointing), and
          - rough uncertainty (standard deviation) of (az, el) adjustment.

    """

    if existing_az_el_adjust is None:
        existing_az_el_adjust = np.zeros((len(ants), 2))

    pointing_offsets = {}
    # Iterate over receptors
    for ant in sorted(ants):
        beams_freq = beams.get(ant.name, [])
        beams_freq = [b for b in beams_freq if b is not None and b.is_valid]
        if not beams_freq:
            continue
        offsets_freq = np.array([b.center for b in beams_freq])
        offsets_freq_std = np.array([b.std_center for b in beams_freq])
        weights_freq = 1. / offsets_freq_std ** 2
        # Do weighted average of offsets over frequency chunks
        results = np.average(offsets_freq, axis=0, weights=weights_freq,
                             returned=True)
        pointing_offset = results[0]
        pointing_offset_std = np.sqrt(1. / results[1])
        logger.debug("%s x=%+7.2f'+-%.2f\" y=%+7.2f'+-%.2f\"", ant.name,
                     pointing_offset[0] * 60, pointing_offset_std[0] * 3600,
                     pointing_offset[1] * 60, pointing_offset_std[1] * 3600)
        # Get existing pointing adjustment
        az_adjust = existing_az_el_adjust[ants.index(ant)][0]
        el_adjust = existing_az_el_adjust[ants.index(ant)][1]
        existing_adjustment = deg2rad(np.array((az_adjust, el_adjust)))
        # Start with requested (az, el) coordinates, as they apply
        # at the middle time for a moving target
        requested_azel = target.azel(timestamp=middle_time, antenna=ant)
        # Correct for refraction, which becomes the requested value
        # at input of pointing model
        rc = RefractionCorrection()

        def refract(az, el):  # noqa: E306, E301
            """Apply refraction correction as at the middle of scan."""
            return [az, rc.apply(el, temperature, pressure, humidity)]
        refracted_azel = np.array(refract(*requested_azel))
        # More stages that apply existing pointing model and/or adjustment
        pointed_azel = np.array(ant.pointing_model.apply(*refracted_azel))
        adjusted_azel = pointed_azel + existing_adjustment
        # Convert fitted offset back to spherical (az, el) coordinates
        pointing_offset = deg2rad(np.array(pointing_offset))
        beam_center_azel = target.plane_to_sphere(
            *pointing_offset, timestamp=middle_time, antenna=ant)
        # Now correct the measured (az, el) for refraction and then apply the
        # existing pointing model and adjustment to get a "raw" measured
        # (az, el) at the output of the pointing model stage
        beam_center_azel = refract(*beam_center_azel)
        beam_center_azel = ant.pointing_model.apply(*beam_center_azel)
        beam_center_azel = np.array(beam_center_azel) + existing_adjustment
        # Make sure the offset is a small angle around 0 degrees
        full_offset_azel = wrap_angle(beam_center_azel - refracted_azel)
        full_adjust_azel = wrap_angle(beam_center_azel - pointed_azel)
        relative_adjust_azel = wrap_angle(beam_center_azel - adjusted_azel)
        # Cheap 'n' cheerful way to convert cross-el uncertainty to azim form
        offset_azel_std = pointing_offset_std / \
            np.array([np.cos(refracted_azel[1]), 1.])
        # We store all variants of the pointing offset since we have it all
        # at our fingertips here
        point_data = np.r_[rad2deg(refracted_azel), rad2deg(full_offset_azel),
                           rad2deg(full_adjust_azel),
                           rad2deg(relative_adjust_azel), offset_azel_std]
        pointing_offsets[ant.name] = point_data
        offsets = point_data.copy()
        offsets[2:] *= 60.
        logger.debug("%s (%+6.2f, %5.2f) deg -> (%+7.2f', %+7.2f')",
                     ant.name, *offsets[[0, 1, 6, 7]])
    return pointing_offsets
