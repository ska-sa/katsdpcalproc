"Tests for pointing.py"

import numpy as np
import katpoint
import pytest
from katsdpcalproc import pointing
import cmath
import random
import scipy.stats

# Creating metadata fpr tests
MIDDLE_TIME = 1691795333.43713
TEMPERATURE = 14.7
HUMIDITY = 29.1
PRESSURE = 897
NUM_CHUNKS = 5
# TODO These tests only use 1 polarization axis
POLS = ["h"]

CENTER_FREQ = 1284000000.0
BANDWIDTH = 856000000.0
N_CHANNELS = 1000

# Conversion factor between standard deviation and
# full width at half maximum (FWHM) of Gaussian beam.
FWHM_SCALE = np.sqrt(8 * np.log(2))


def _pointing_offsets(max_extent, num_pointings):
    """Build up sequence of pointing offsets running linearly in x and y directions."""
    scan = np.linspace(-max_extent, max_extent, num_pointings // 2)
    offsets_along_x = np.c_[scan, np.zeros_like(scan)]
    offsets_along_y = np.c_[np.zeros_like(scan), scan]
    return np.r_[offsets_along_y, offsets_along_x]


# Calculating channel and chunk freqencies
channel_freqs = CENTER_FREQ + (np.arange(N_CHANNELS) - N_CHANNELS / 2) * (BANDWIDTH / N_CHANNELS)
chunk_freqs = channel_freqs.reshape(NUM_CHUNKS, -1).mean(axis=1)
target = katpoint.Target(
    body="J1939-6342, radec bfcal single_accumulation, 19:39:25.03, -63:42:45.6")
# Maximum distance of offset from target, in degrees
offsets = _pointing_offsets(max_extent=1.0, num_pointings=8)

# Creating list of antenna objects
ANT_DESCRIPTIONS = ["""m000,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:20.6 0 0:01:14.2 0:02:58.5 0:00:05.1 0:00:00.4
                          0:20:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22""",
                    """m001,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:15.6 0 0:01:09.2 0:01:58.5 0:00:05.1 0:00:00.4
                          0:16:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22"""]


ants = [katpoint.Antenna(ant) for ant in ANT_DESCRIPTIONS]
az_el_adjust = np.zeros((len(ants), 2))


def generate_bp_gains(offsets, ants, channel_freqs, pols, beam_center=(0.5, 0.5)):
    """ Creating bp_gains, numpy array of shape:
        (no.offsets, no.freqs, no.polarisations, no.antennas)"""
    bp_gains = np.zeros(
        (len(offsets),
         len(channel_freqs),
         len(pols),
         len(ants)),
        dtype=np.complex64)
    for i in range(len(offsets)):
        for f in range(len(channel_freqs)):
            for pol in range(len(pols)):
                ex_width = []
                for a, ant in enumerate(ants):
                    # expected widths for each frequency channel
                    expected_width = pointing._voltage_beamwidths(
                        ant.beamwidth, channel_freqs[f], ant.diameter)
                    ex_width.append(expected_width)

                gains = []
                for k in ex_width:
                    new_beam = pointing.BeamPatternFit(beam_center, k, 1.0)
                    g = new_beam(x=offsets[i].T)
                    cmplx = random.uniform(-1.5, 1.5)
                    g = cmath.rect(g, cmplx)
                    gains.append(np.array(g))
                gains = np.array(gains).T
                bp_gains[i][f][pol] = gains

    return bp_gains


bp_gains = generate_bp_gains(offsets, ants, channel_freqs, POLS)
data_points = pointing.get_offset_gains(
    bp_gains,
    offsets,
    ants,
    channel_freqs,
    POLS,
    NUM_CHUNKS)
beams = pointing.beam_fit(data_points, ants, NUM_CHUNKS)
pointing_offsets = pointing.calc_pointing_offsets(
    ants,
    MIDDLE_TIME,
    TEMPERATURE,
    HUMIDITY,
    PRESSURE,
    beams,
    target,
    az_el_adjust)


def test_get_offset_gains_len():
    """Test that length of data_points equals the legnth of antenna list"""
    assert len(data_points) == len(ants)


def test_get_offset_gains_shape():
    """Test that incorrect shape of bp_gains will ValueError"""
    with pytest.raises(ValueError):
        pointing.get_offset_gains(
            bp_gains[0],
            offsets,
            ants,
            channel_freqs,
            POLS,
            NUM_CHUNKS)


def test_get_offset_gains_len2():
    """Test legnth of each data_points element"""
    for i in range(len(data_points)):
        assert len(list(data_points.items())[i][1]) == NUM_CHUNKS * len(offsets)
        for j in range(NUM_CHUNKS * len(offsets)):
            assert len(list(data_points.items())[i][1][j]) == 5


def test_beam_fit_type():
    """Testing that the output of beam_fit are of type BeamPatternFit"""
    assert len(beams) == len(ants)
    for i in ants:
        assert beams[i.name][0] is None and beams[i.name][-1] is None
        for j in range(1, NUM_CHUNKS - 1):
            assert isinstance(beams[i.name][j], pointing.BeamPatternFit)


def test_calc_pointing_offsets_len():
    """Test that the legnth of each pointing offset solution
       =10 (5 sets of (x,y) coordinates)"""
    assert len(pointing_offsets) == len(ants)
    for i in range(len(pointing_offsets)):
        assert len(list(pointing_offsets.items())[i][1]) == 10


def test_fit_primary_beams():
    """Compare widths and beam center of simulated primary beams
       from beam_fit and original beam object"""
    expected_widths = {}
    compare_beam_center = {}
    for a, ant in enumerate(ants):
        ex_width = []
        beam_center = []

        for chunk in range(NUM_CHUNKS):
            expected_width = pointing._voltage_beamwidths(
                ant.beamwidth, chunk_freqs[chunk], ant.diameter)
            ex_width.append(expected_width)
            expected_widths[ant.name] = ex_width
            beam = pointing.BeamPatternFit((0.5, 0.5), expected_width, 1.0)
            beam_center.append(beam)
            compare_beam_center[ant.name] = beam_center

    # Feeding simulated data_points into beam_fit function
    beams = pointing.beam_fit(data_points, ants, NUM_CHUNKS)
    # Comparing output of beam_fit to the original beam object
    # Testing the expected beam center and comparing expected widths
    for chunk in range(1, NUM_CHUNKS - 1):
        for ant in beams.keys():
            assert beams[ant][chunk].expected_width == pytest.approx(
                expected_widths[ant][chunk], abs=0.0001)
            assert beams[ant][chunk].center == pytest.approx(
                compare_beam_center[ant][chunk].center, abs=0.001)


def _beam_params(beam):
    """Extract beam parameters [A, x0, y0, fwhm_x, fwhm_y] as a vector."""
    return np.r_[beam.height, beam.center, beam.width]


def _beam_params_std(beam, error_scale):
    """Extract stdev of beam parameters [A, x0, y0, fwhm_x, fwhm_y] as vector."""
    return np.r_[beam.height, beam.width / FWHM_SCALE, beam.width] * error_scale


def test_gaussian_beam_fit_accuracy():
    r"""Reproduce Jim Condon's 2-D Gaussian fitting simulation.

    This repeats a computer simulation described in [Condon1997]_, which
    estimates the uncertainty on the fitted parameters of a 2-D elliptical
    Gaussian fitted to samples on a regular square grid of "pixels". This
    function goes a step further and tests the residuals for normality as
    a unit test.

    Notes
    -----
    The variable names try to stick to the notation of the paper for clarity.

    Quoting from the [Condon1997]_ paper::

      I generated an empty 1024 x 1024 pixel image and inserted Gaussian
      pseudorandom noise with rms amplitude :math:`\mu` into each pixel of
      area :math:`h^2`. Then 1000 elliptical Gaussians with amplitude
      :math:`A = 20\mu`, major diameter :math:`\theta_M = 4h`, minor diameter
      :math:`\theta_m = 10h/3`, and position angle :math:`\phi = 45^{\circ}:
      were added to the image. The 1000 sets of parameters :math:`A`,
      :math:`x_0`, :math:`y_0`, :math:`\theta_M`, :math:`\theta_m`, and
      :math:`\phi` were extracted by Gaussian fitting. Figure 1 shows
      histograms of the differences between the fitted and input parameters,
      normalized by the theoretical rms errors, which were obtained by
      inserting the fitted (rather than the normally unknown input) parameter
      values into Eqs. (20) and (21). These histograms match the calculated
      unit Gaussians (continuous curves) within the expected sampling errors.

    References
    ----------
    .. [Condon1997] J. J. Condon, "Errors in Elliptical Gaussian Fits,"
       PASP, vol. 109, pp. 166-172, Feb 1997.
    """
    # The number of Gaussians to fit (reduced from 1000 to save time)
    T = 100
    # The size of a pixel in data coordinates (i.e. length of each pixel side)
    h = 0.1
    # The rms noise amplitude on each pixel
    mu = 0.01
    # The pixel locations along each axis (x or y), wide enough to cover beam
    samples_1d = h * np.arange(-30, 30)
    # Generate square "image" on which Gaussians will be sampled
    xy_k = np.reshape(np.meshgrid(samples_1d, samples_1d), (2, -1))
    # Generate beam centers within a few pixels of origin
    x0, y0 = 10 * h * (np.random.rand(2, T) - 0.5)
    # Beam amplitude
    A = 20 * mu
    # Beam diameters along x and y
    # XXX We had to double the width to avoid excessive fitting failures
    theta_M = 8 * h
    theta_m = 20 * h / 3
    # Vary the beam diameter by a few percent to make it more realistic
    fwhm_x = theta_M * (1 + 0.04 * (np.random.rand(T) - 0.5))
    fwhm_y = theta_m * (1 + 0.04 * (np.random.rand(T) - 0.5))
    # Beam position angle is set to phi = 0 instead due to fitter limitation
    # Effective number of independent samples in fitted Gaussian
    N = (np.pi / FWHM_SCALE ** 2) * fwhm_x * fwhm_y / h ** 2
    # Overall signal-to-noise (voltage) ratio of the Gaussian fit
    rho = np.sqrt(N) * (A / mu)
    # Overall scale factor for parameter errors
    error_scale = np.sqrt(2) / rho
    # Normalized differences between fitted and true parameters / uncertainties
    p_errors = np.full((10, T), np.nan)
    # Iterate over Gaussians to fit
    for t in range(T):
        # XXX This assumes that BeamPatternFit fits a Gaussian beam
        true_beam = pointing.BeamPatternFit((x0[t], y0[t]), (fwhm_x[t], fwhm_y[t]), A)
        # Generate noisy beam amplitude samples on pixel grid
        a_k = true_beam(xy_k) + mu * np.random.randn(xy_k.shape[1])
        # Assume we have a good idea of beamwidth (needed to check beam validity)
        beam = pointing.BeamPatternFit((0.0, 0.0), (theta_M, theta_m), max(a_k))
        beam.fit(xy_k, a_k, mu)
        if not beam.is_valid:
            continue
        # Theoretical rms errors (based on fitted parameters)
        beam_std = _beam_params_std(beam, error_scale[t])
        # Compare estimated / fitted to true parameters and standardise the errors
        p_errors[:5, t] = (_beam_params(beam) - _beam_params(true_beam)) / beam_std
        # Estimated beam parameter uncertainties from fitter
        beam_std_estm = np.r_[beam.std_height, beam.std_center, beam.std_width]
        # Theoretical rms errors (based on true parameters)
        true_beam_std = _beam_params_std(true_beam, error_scale[t])
        # XXX Estimate the uncertainty of the uncertainty (beware, thumbsuck!)
        beam_std_std = beam_std * error_scale[t]
        beam_std_std[0] *= 1.1
        beam_std_std[1:] *= np.sqrt(FWHM_SCALE)
        # Compare estimated / fitted to true uncertainties and standardise errors
        p_errors[5:, t] = (beam_std_estm - true_beam_std) / beam_std_std
    # Remove failed fits
    p_errors = p_errors[:, np.isfinite(p_errors[0])]
    # Check the success rate of the fits
    assert p_errors.shape[1] >= np.floor(0.995 * T)
    # Check 5 parameters and 5 uncertainties individually
    for p_error in p_errors:
        # Check that errors are standard normal (mean=0, std=1).
        # The threshold is 1 - 0.95 ** 0.05 to ensure a 5% chance of a random
        # assert failure when doing 20 checks (two checks each on 10 variables).
        assert scipy.stats.kstest(p_error, scipy.stats.norm.cdf).pvalue > 0.0026
        # Check that errors are normal (more stringent on outliers)
        assert scipy.stats.shapiro(p_error).pvalue > 0.0026
