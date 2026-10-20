"""Pytest-style tests for secant-based delay solving."""

import numpy as np
from katsdpcalproc.delay import _secant_fast, fft_secant


def wrap_angle(th):
    return (th + np.pi) % (2.0 * np.pi) - np.pi


def test_fft_secant_estimates(make_delay_data, calculate_params, observation_params, random):
    """Test that the fft_secant solver recovers known phase slopes."""
    true_freq = np.array([[0.15, -0.30], [0.6, -1.2]], dtype=np.float64)
    chans, _, noise_var, _ = calculate_params(
        observation_params["sample_rate"], observation_params["N_CHANS"],
        observation_params["dump_period"], observation_params["flux"],
        observation_params["sefd"])
    x = make_delay_data(true_freq.reshape(-1), chans, ampl=observation_params["flux"],
                        noise_var=noise_var, rng=random, add_spike=True)
    est = fft_secant(x, NFFT=2 * observation_params["N_CHANS"]).reshape(true_freq.shape)
    assert est.shape == true_freq.shape
    np.testing.assert_allclose(est, true_freq, atol=0.1)


def test_fft_secant_estimates_noisy(make_delay_data, calculate_params, observation_params, random):
    """Test that the fft_secant solver recovers known phase slopes in noisier conditions."""
    true_freq = np.array([[0.15, -0.30], [0.6, -1.20]], dtype=np.float64)
    chans, _, noise_var, snr = calculate_params(
        observation_params["sample_rate"], observation_params["N_CHANS"],
        observation_params["dump_period"], observation_params["flux"],
        observation_params["sefd"])
    # lower the flux amplitude to increase noise of data
    x = make_delay_data(true_freq.reshape(-1), chans, ampl=0.01,
                        noise_var=noise_var, rng=random, add_spike=False)
    est = fft_secant(x, NFFT=2 * observation_params["N_CHANS"]).reshape(true_freq.shape)
    assert est.shape == true_freq.shape
    np.testing.assert_allclose(est, true_freq, atol=1e-2)


def test_secant_on_steep_slopes(observation_params):
    """Test the secant solver on phase slopes near the FFT alias boundary."""
    n_chans = observation_params["N_CHANS"]
    nfft = 2 * n_chans
    steep_slopes = 2 * np.pi * np.array([n_chans - 0.45, n_chans, n_chans + 0.45]) / nfft
    steep_signals = np.exp(1j * np.outer(steep_slopes, np.arange(n_chans)))
    steep_slope_estimates = fft_secant(steep_signals, NFFT=nfft)
    residuals = wrap_angle(steep_slope_estimates - steep_slopes)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


def test_secant_fast(make_delay_data, calculate_params, observation_params, random):
    """Test slope recovery and unconverged behavior of secant_fast."""
    n_chans = observation_params["N_CHANS"]
    nfft = 2 * n_chans
    true_phase_slope = np.array([0.44], dtype=np.float64)
    chans, _, noise_var, snr = calculate_params(
        observation_params["sample_rate"], n_chans, observation_params["dump_period"],
        observation_params["flux"], observation_params["sefd"])
    x = make_delay_data(true_phase_slope, chans, ampl=observation_params["flux"],
                        noise_var=noise_var, rng=random, add_spike=True)
    coarse_bin_width = 2 * np.pi / nfft
    left = true_phase_slope - coarse_bin_width
    right = true_phase_slope + coarse_bin_width

    est = _secant_fast(x, left, right, epsilon=1e-10, max_iters=100,
                       discard_unconverged=False)
    np.testing.assert_allclose(est, true_phase_slope, atol=0.01)

    unconverged = _secant_fast(x, left, right, epsilon=1e-10, max_iters=0,
                               discard_unconverged=True)
    assert np.isnan(unconverged[0])


def test_secant_fast_flagged_masked_channels(
        make_delay_data, calculate_params, observation_params, random):
    """Test masked channels should still yield a usable secant estimate."""
    n_chans = observation_params["N_CHANS"]
    true_phase_slope = np.array([0.44], dtype=np.float64)
    chans, delay_alias, noise_var, snr = calculate_params(
        observation_params["sample_rate"], n_chans, observation_params["dump_period"],
        observation_params["flux"], observation_params["sefd"])
    x = make_delay_data(true_phase_slope, chans, ampl=observation_params["flux"],
                        noise_var=noise_var, rng=random)
    channel_mask = np.ones(n_chans, dtype=np.float64)
    channel_mask[10:900] = np.nan
    x *= channel_mask[np.newaxis, :]
    # Bracket must stay inside the main lobe; the gap creates sidelobes the secant can lock onto
    coarse_bin_width = 2 * np.pi / (2 * n_chans)
    left = true_phase_slope - coarse_bin_width
    right = true_phase_slope + coarse_bin_width
    est = _secant_fast(x, left, right, epsilon=1e-10, max_iters=100,
                       discard_unconverged=True)
    np.testing.assert_allclose(est, true_phase_slope, atol=0.01)
